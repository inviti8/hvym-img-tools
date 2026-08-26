# AUTH.md — keeping the GPU to ourselves

The server runs inference on a GPU that **bills per second**. An open endpoint is not
just a data-exposure question, it is a *spend* question: anyone who finds the URL can run
up the bill. This is the minimum viable gate for the Inkternity demo.

## What this is, and what it is not

A **shared API key**, checked in constant time, on every `/tools/*` route.

**It is not an identity boundary.** A key shipped inside a distributed desktop app is
extractable — anyone can pull it from the binary, or read it off the wire with a
TLS-intercepting proxy. Anyone determined to get the key will get the key.

What it does buy, which is real and worth having:

- keeps drive-by scanners, crawlers and accidental traffic off a metered GPU
- makes "who may call this" an explicit, tested decision rather than an oversight
- gives a **revocation lever** — rotate the key, ship a client update
- costs essentially nothing to run and nothing to reason about

Treat it as a lock on a door, not a vault. Size the exposure accordingly: the endpoint
takes a drawing and returns a mesh, so the worst case of a leaked key is **someone
spending your GPU budget**, not data loss.

## Using it

**Server** — set one or more keys (comma-separated) and it turns on automatically:

```sh
export HVYM_API_KEY="$(python -m hvym_img_tools.core.auth)"   # generates a strong key
uv run hvym-img-serve
```

With `HVYM_API_KEY` unset, auth is **disabled** and the server logs a loud warning at
startup. That is fine for local development and wrong for anything reachable.

**Client** — either header works:

```sh
curl -X POST https://host/tools/reangle \
     -H "X-API-Key: $HVYM_API_KEY" \
     -F image=@drawing.png -o char.glb

curl -H "Authorization: Bearer $HVYM_API_KEY" ...   # equivalent
```

| Route | Auth |
|---|---|
| `POST /tools/{name}` | **required** |
| `GET /tools` | **required** |
| `GET /healthz` | **open** — orchestrator probes carry no credentials, and it costs no GPU |

Rejections are always `401` with an identical body whether the key was missing or wrong,
so the endpoint is not an oracle for guessing.

## Rotating a key

`HVYM_API_KEY` accepts a comma-separated list, so rotation needs no downtime:

1. `HVYM_API_KEY="newkey,oldkey"` — restart; both work
2. Ship the client update using `newkey`
3. `HVYM_API_KEY="newkey"` — restart; `oldkey` is dead

## Operational notes

- **Always use HTTPS.** Over plain HTTP the key is in cleartext on the wire. RunPod's
  proxy endpoints are HTTPS by default; do not terminate TLS yourself without a reason.
- **Never commit a key.** `.env` is gitignored; keep it that way.
- **Do not log it.** The server never logs key material, only whether auth is enabled.
- **The result cache is shared across callers.** Cache keys are `sha256(image + params)`,
  so two callers sending the same drawing share a cache entry. That is the intended
  behaviour (it is what makes re-requests free) but it means the cache is not a
  per-tenant store. Do not put anything tenant-scoped in it without changing the key.

## Upgrade path

When the demo becomes a product, in rough order of effort:

1. **Rate limiting / quota per key** — the cheapest next win, and the one that directly
   caps spend. A leaked key then costs a bounded amount rather than an unbounded one.
2. **Per-user tokens issued at runtime** — the app authenticates a user, the server mints
   a short-lived token. The long-lived secret stops living in the binary.
3. **A thin authenticating proxy** in front of the GPU service, holding the service key
   server-side so it never ships to clients at all.
4. **Request signing** (HMAC over body + timestamp + nonce) if replay becomes a concern.

Steps 2 and 3 are what turn this from a spend-control measure into an actual identity
boundary. Until then, be honest in any threat modelling that it is the former.
