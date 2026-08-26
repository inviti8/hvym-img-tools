# hvym-img-tools

A suite of **self-hosted, AI-augmented image tools** for creative apps — served over a
clean HTTP contract (and CLIs) so a native app can call them without depending on any
third-party AI cloud. **Sovereign** (self-hostable, permissive licenses) and **modular**
(each capability is a self-contained *tool* that plugs into a shared core).

**First client:** [Inkternity](../infinipaint) (a C++ drawing app).
**First tool:** **reangle** — style-preserving camera-angle adjustment of a character
drawing (build a rough 3D proxy, re-project the artist's *original* linework, turn the
camera a little; the drawing is moved, never regenerated).

## Status

The cost-model gate is settled and the framework is built.

| | |
|---|---|
| **Reconstruction** | **0.114 s** (TripoSR) vs ~8–10 min for the prototype — ~340× end to end |
| **Request, warm** | **1.99 s** on the live serverless endpoint; **0.02 s** on a cache hit |
| **Cold start** | **~260 s** for a fresh worker to pull the 6.48 GB image (~48 s if the host has it) |
| **Cost** | **$0.00054/image** → ~$0.55/mo compute + $0.70/mo cache volume at 1,000 drawings |
| **Licensing** | **MIT end to end** — no CC-BY-NC anywhere |
| **Verdict** | 🟢 **GREEN** — see [`docs/BENCHMARK.md`](docs/BENCHMARK.md) |

## Quick start

```sh
uv sync --extra dev --extra server
uv run pytest                       # CPU-only, no GPU or weights needed
uv run hvym-img-serve               # http://localhost:8000/docs
```

```sh
# one drawing in, one textured .glb out
curl -X POST http://localhost:8000/tools/reangle \
     -H "X-API-Key: $HVYM_API_KEY" \
     -F image=@drawing.png -F mc_resolution=256 -o char.glb

uv run hvym-img reangle --in drawing.png --out char.glb   # same, locally
```

## Architecture

- **core** — FastAPI server, tool **registry** (auto-mounts tools at `POST /tools/{name}`),
  a shared **model cache** (warm GPU models once), image/3D utilities, input-hash result
  cache. Deliberately **CPU-importable**: neither torch nor FastAPI is imported at load.
- **tools/** — one package per capability implementing a small `Tool` interface. Adding a
  tool touches nothing else — it inherits the endpoint, OpenAPI, caching and a CLI for free.
- **one GPU container** serves every tool; deploy to RunPod serverless or a persistent box.

## Deployment

**Live:** endpoint `69j3vhp0el0wv0` (RTX 4090 / L4, EU-RO-1), images on
[GHCR](https://github.com/inviti8?tab=packages), result cache on a shared network volume
so it survives scale-to-zero.

RunPod Serverless (scale-to-zero) behind a small **authenticating proxy** that keeps the
RunPod key server-side — an account key grants full account access and must never ship in a
desktop client. The proxy mirrors the server's HTTP contract exactly, so client code is
identical either way. See [`docs/DEPLOY.md`](docs/DEPLOY.md) and
[`docs/AUTH.md`](docs/AUTH.md).

## Docs

| | |
|---|---|
| [`AGENTS.md`](AGENTS.md) | **start here** — onboarding, the `Tool` contract, how to add a tool |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | measured latency, cost model, the GREEN decision, and the gotchas |
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | images, serverless endpoint, proxy, cold start |
| [`docs/WARMING.md`](docs/WARMING.md) | the demo warm switch vs. the product's client lease, and what warm costs |
| [`docs/AUTH.md`](docs/AUTH.md) | API-key scheme, threat model, upgrade path |
| [`docs/tools/reangle.md`](docs/tools/reangle.md) | the reference tool |

Reangle's authoritative pipeline spec lives in the client repo:
[`../infinipaint/docs/design/REANGLE_PIPELINE.md`](../infinipaint/docs/design/REANGLE_PIPELINE.md).

## Conventions

- Python via **`uv`** (never raw pip/python).
- Sovereign: self-hostable weights only; **document each tool's model licenses** (CC-BY-NC
  is demo-only, never a shipped default).
- Keep `core` generic and CPU-importable; heavy ML stays inside the tool.
