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
| `mc_resolution` | int 64–512 | `256` | Marching-cubes grid — the main cost/quality lever |
| `backbone` | str | `triposr` | Reconstruction backbone |

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
