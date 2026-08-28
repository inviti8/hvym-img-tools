# CLIENT.md — integrating Inkternity against this API

Everything a client needs to call the service correctly. The wire contract is
**identical** whether you point at the authenticating proxy (shipped shape) or
at `core.server` on a persistent pod, so build against this once.

> **Read [Timeouts](#timeouts-the-one-thing-that-will-bite-you) before writing the
> HTTP call.** A default HTTP timeout fails *every* cold start. It is the single
> most common way this integration goes wrong.

## The call

```
POST {base}/tools/reangle
X-API-Key: <scoped key>
Content-Type: multipart/form-data
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `image` | file | *required* | Character drawing, any size or background. Max **32 MB** |
| `mc_resolution` | int 64–512 | `256` | Marching-cubes grid. **TripoSR only** — TRELLIS ignores it |
| `backbone` | str | `trellis` | `trellis` (sounder geometry) or `triposr` (tighter silhouette) |
| `target_faces` | int 2 000–200 000 | *backbone's own* | Decimation target. Unset: TRELLIS caps at 20 000, TripoSR is uncapped |
| `seed` | int 0–2³¹-1 | `0` | Only TRELLIS is stochastic. Fixed so results stay cacheable |

**The default backbone changed in reangle 0.3.0** (was `triposr`). Results are
cached by `sha256(image + params + tool version)`, and the version bump means no
0.2.0 result is served for a 0.3.0 request — the first call per drawing is a miss.
A worker rejects a backbone it has no weights for rather than substituting one.

Auth accepts either `X-API-Key: <key>` or `Authorization: Bearer <key>`. Keys are
compared in constant time; minimum length 16.

**Success** is `200` with the raw `.glb` as the body:

| | |
|---|---|
| `Content-Type` | `model/gltf-binary` |
| `Content-Disposition` | `attachment; filename="char.glb"` |
| `X-Cache` | `HIT` or `MISS` |
| `X-Tool-Version` | tool version that produced it |
| `X-Upstream-Elapsed` | seconds of actual work |

The body is **binary, not JSON**. Errors *are* JSON (`{"detail": "..."}`), so
branch on status code, not on content type.

```sh
curl -X POST "$BASE/tools/reangle" \
     -H "X-API-Key: $KEY" \
     -F image=@drawing.png -F mc_resolution=256 \
     --max-time 300 -o char.glb
```

## Timeouts: the one thing that will bite you

**Set the request timeout to at least 300 s.** Not a suggestion:

| Situation | Time to first byte |
|---|---|
| Cache hit | ~0.02 s of work |
| Warm worker, new drawing | ~2 s of work |
| Cold worker, host has the image | ~48 s |
| **Cold worker, fresh host** | **up to ~260 s** |

A cold worker must pull a 6.48 GB image before it loads a model
(BENCHMARK.md §6b). libcurl's defaults, most HTTP wrappers' 30 s, and anything
"reasonable" will abort mid-cold-start and surface as a network error — while the
job keeps running and completes server-side.

Practical shape for the UI: show a spinner that survives minutes, and say
*"warming up — first request after idle takes a few minutes"* rather than showing
a progress bar you cannot honestly fill. During a demo the operator can remove
the wait entirely with `scripts/warm.py on` ([WARMING.md](WARMING.md)).

## Status codes

| Code | Meaning | What the client should do |
|---|---|---|
| `200` | success, binary body | use it |
| `401` | missing/invalid key | surface as config error — do **not** retry |
| `413` | image over 32 MB | downscale and resubmit |
| `422` | malformed form fields | fix the request — a bug, not a transient |
| `500` | the tool itself failed | show `detail`; retrying rarely helps |
| `502` | upstream/worker problem | retry once after a short pause |
| `503` | proxy not configured | server-side misconfig — surface, don't retry |
| `504` | exceeded the proxy's budget | the job may still finish; retry re-uses the cache |

## Warming

Cold start is the sharpest edge of this deployment (up to ~260 s, see above). The
service lets a client hold a **lease** that keeps a GPU worker awake, so an artist
who flips on "enable inference" pays the wait once instead of on every idle gap.

```
POST   /warm     X-API-Key: <key>   {"lease_id": "<opaque>", "label": "<optional>"}
GET    /warm                        (no key required)
DELETE /warm     X-API-Key: <key>   {"lease_id": "<opaque>"}
```

`POST /warm` acquires on first call and extends thereafter. Omit `lease_id` on the
first call and the server issues one; send that same id back every time.

```json
{"lease_id": "9f2c...", "state": "warming", "ready": false,
 "elapsed_s": 12.3, "expires_at": "2026-08-27T00:14:02+00:00",
 "lease_ttl_s": 60.0, "renew_within_s": 20.0, "active_leases": 1}
```

| Field | Use |
|---|---|
| `state` | `cold` / `warming` / `warm` — drive the indicator from this |
| `ready` | `true` only when a worker can serve *now* |
| `elapsed_s` | seconds since warming began — show it, don't compute an ETA |
| `renew_within_s` | **renew at least this often**, or the lease lapses |
| `lease_ttl_s` | how long the current renewal bought |

### The renew-within contract

**Re-POST every `renew_within_s` (20 s by default) for as long as the toggle is
on.** The lease lives `lease_ttl_s` (60 s), so two consecutive missed renewals are
survivable, and a client that crashes, sleeps, or loses the network stops paying
within about a minute without doing anything. That asymmetry is the point: silence
means release.

**The renewal poll is also the notification channel.** Every response carries the
current state, so the UI needs no push, no WebSocket, and no reconnect logic — poll
to hold the lease, and render whatever comes back. If the poll itself fails, show
cold; it probably is.

**Do not build your own keepalive.** The proxy pings the GPU internally every ~6 s,
which is under the endpoint's 10 s idle timeout — that cadence is the server's
business and is not something the client should mirror or tune. Your only job is to
renew the lease.

**Release explicitly** with `DELETE /warm` when the artist turns the toggle off, or
on clean shutdown. It is idempotent, so a retry after a dropped response is safe.
Not releasing is not a disaster — the lease lapses — but it wastes up to a minute
of GPU time.

### Notes

- `/warm` uses the **same scoped key** as `/tools/reangle`. There is no second
  credential, and no path here reaches the RunPod account key.
- `GET /warm` is unauthenticated: it spends nothing and cannot start a worker, so
  the UI can show a state indicator before the artist opts in.
- Leases are **refcounted**. Two Inkternity instances behind one proxy share a
  single warm worker rather than paying twice.
- Against a **persistent-box deployment** (`core.server`, no scale-to-zero) `/warm`
  is a truthful no-op returning `state: "warm", ready: true, no_op: true`. Hold a
  lease unconditionally and the same client code works against either deployment.
- **Warm time is billable** — it is the metered unit in the paid product, so treat
  the toggle as something the artist turns on deliberately, not a default-on.

## Retries are cheap and safe

Results are cached by `sha256(image + params)`, so **the same drawing and params
always produce the same result**, and a repeat costs ~0.02 s instead of ~2 s.
That makes retry-on-failure genuinely safe: a retry after a timeout usually hits
a completed result rather than redoing the work.

It also sets the intended access pattern: **one call per drawing**, then all
interaction — orbiting, re-projection, tweaking — is local in Inkternity. The
service is not in the interactive loop.

## Do not talk to RunPod directly

Point only at the proxy. Calling RunPod Serverless directly requires a RunPod
*account* key, which grants full account access — creating pods, spending the
balance, deleting resources — and must never ship in a desktop binary. The proxy
exists to hold that key. See [AUTH.md](AUTH.md).

If you ever do bypass it, note that `/runsync` caps at ~90 s server-side and then
returns the job still `IN_QUEUE`; you must poll `/status/{id}` from there or every
cold start looks like a failure. The proxy already handles this.

## A note on OpenAPI

The **direct server** (`core.server`) publishes an accurate schema at `/docs`,
generated from each tool's input model — useful for exploring.

The **proxy does not**. It forwards `POST /tools/{name}` generically, so its
generated spec has no request body. This document is the contract for the proxy;
do not codegen a client from its `/openapi.json`.

## Running it locally

Against the live serverless endpoint, with the RunPod key kept local:

```sh
HVYM_API_KEY=<scoped key> \
RUNPOD_API_KEY=<runpod key> \
RUNPOD_ENDPOINT_ID=<endpoint id> \
HVYM_PROXY_TIMEOUT=600 \
uv run hvym-img-proxy            # http://localhost:8080
```

`GET /healthz` is unauthenticated and reports whether auth and RunPod are
configured, never the values — use it as a readiness probe.

For CPU-only client development with no GPU and no RunPod account, run the real
server locally; the contract above is identical, just slower:

```sh
uv run hvym-img-serve            # http://localhost:8000/docs
```

## Checklist before wiring the UI

- [ ] Timeout ≥ 300 s on the request
- [ ] `X-API-Key` sent on every call; key read from config, never hardcoded
- [ ] Response handled as **binary**, errors parsed as JSON by status code
- [ ] A "first call may take minutes" affordance in the UI
- [ ] `401` and `503` surfaced as configuration problems, not transient failures
- [ ] Base URL configurable — proxy today, possibly a pod later
