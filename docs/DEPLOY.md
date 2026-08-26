# DEPLOY.md — serverless GPU + authenticating proxy

The shipped shape (**option C**): RunPod Serverless does the GPU work and scales to
zero; a small always-on proxy holds the RunPod key and authenticates Inkternity.

```
  Inkternity                proxy (no GPU)              RunPod Serverless
  ──────────                ──────────────              ─────────────────
  POST /tools/reangle  ──▶  X-API-Key check       ──▶   worker pulls job
  X-API-Key: <scoped>       base64 the image            matte → TripoSR →
                            RUNPOD_API_KEY stays        UV bake → .glb
  ◀── char.glb ─────────    server-side           ◀──   base64 out
                                                        scale-to-zero when idle
```

## Why the proxy is not optional

Calling RunPod Serverless directly means embedding a **RunPod API key** in the
desktop app. A RunPod *account* key can create pods, spend the balance and delete
resources — so a leaked one costs far more than a leaked scoped key, which can
only ask for a mesh. The proxy is what keeps that key off client machines.

It also **mirrors the direct server's HTTP contract exactly** — same
`POST /tools/{name}`, same multipart in, same binary out, same `X-Cache` and
`X-Tool-Version` headers. Inkternity's client code is therefore identical whether
it points at this proxy or at a persistent pod running `core.server`, so the
deployment can change later without touching the client.

Read [`AUTH.md`](AUTH.md) for the threat model — the scoped key is spend control
and a revocation lever, not an identity boundary.

## Images

| Image | Role | Contents |
|---|---|---|
| `docker/Dockerfile` | GPU worker | torch, TripoSR, torchmcubes, **weights baked in** |
| `docker/Dockerfile.proxy` | proxy | fastapi + httpx only — no torch, no CUDA |

The GPU image is multi-stage: CUDA `devel` compiles `torchmcubes`, then the wheel
is copied into a `python:3.10-slim` runtime, so **nvcc never ships**. torch's pip
wheels bundle their own CUDA libs, so the runtime needs only the host driver.

Weights (isnet ~176 MB, TripoSR ~1.7 GB) are **baked in**. Downloading them on a
cold worker would add ~1.9 GB before the first job — exactly what FlashBoot exists
to avoid.

### Build-time assertions

The build fails rather than shipping a subtly broken image if:

- `onnxruntime-gpu` lacks `CUDAExecutionProvider` — `rembg` (a TripoSR import)
  drags in **CPU onnxruntime**, which shadows the GPU build and silently makes the
  matte **6.4 s instead of 0.03 s**, with no error
- `torchmcubes` will not import
- the isnet download is truncated
- `reangle` fails to register
- (proxy) anything makes it import `torch`

## Build and push

**Build in CI, not on a workstation.** The GPU image is ~7-8 GB and pulls ~5 GB
of torch/CUDA wheels; a runner sits on the same network as GHCR, so the push is
effectively free, whereas a local build moves that volume twice over whatever
uplink the developer has.

```sh
gh workflow run images.yml -f tag=0.1.0        # .github/workflows/images.yml
gh run watch $(gh run list --workflow=images.yml --limit 1 --json databaseId -q '.[0].databaseId')
```

Locally, if you need to reproduce a build failure:

```sh
docker build -f docker/Dockerfile       -t hvym-img-tools:0.1.0 .
docker build -f docker/Dockerfile.proxy -t hvym-img-proxy:0.1.0 .
```

The GPU image builds for compute **8.6 and 8.9** (`TORCH_CUDA_ARCH_LIST`), covering
A40/A6000/A5000/3090 and 4090/L4/L40S — the serverless tiers we might land on.

## RunPod Serverless endpoint

Create it against the GPU image with:

| Setting | Value | Why |
|---|---|---|
| Container image | `<registry>/hvym-img-tools:0.1.0` | |
| Container start command | *(default)* → `serverless` | entrypoint's default mode |
| GPU tier | 24 GB | peak VRAM measured at **4.44 GB** — no need to pay for 48 GB |
| Max workers | start at 1–2 | one request per drawing; scale later |
| Idle timeout | 5–10 s | scale-to-zero is the whole point |
| FlashBoot | on | mitigates cold start |
| `HVYM_CACHE_DIR` | a network volume | otherwise the cache dies with the worker |

**The result cache is per-worker unless you mount shared storage.** Cache keys are
`sha256(image + params)`, so a network volume lets a second worker serve a drawing
the first one already built. Without it, every cold worker starts cold-cached and
the "instant re-request" property is lost.

## Running the proxy

```sh
docker run -d --restart=unless-stopped -p 8080:8080 \
  -e HVYM_API_KEY="$(python -m hvym_img_tools.core.auth)" \
  -e RUNPOD_API_KEY=...      `# never leaves the server` \
  -e RUNPOD_ENDPOINT_ID=...  \
  <registry>/hvym-img-proxy:0.1.0
```

Then Inkternity talks only to the proxy:

```sh
curl -X POST https://proxy.example.com/tools/reangle \
     -H "X-API-Key: $SCOPED_KEY" \
     -F image=@drawing.png -F mc_resolution=256 -o char.glb
```

**Put TLS in front of the proxy.** Over plain HTTP the scoped key is cleartext on
the wire.

## Cold start

**Measured on the live endpoint: ~260 s (4.4 min) onto a host that does not yet
have the image; ~48 s onto one that does.** Model load is only 13.7 s of that —
the rest is pulling 6.48 GB, of which a single torch+CUDA layer is 3.96 GB.
FlashBoot does not help the first pull onto a given host. See BENCHMARK.md §6b.

This is the sharpest edge of the deployment. A demo that opens with a 4-minute
wait reads as broken, so pick a mitigation deliberately:

| Mitigation | Cost | Effect |
|---|---|---|
| FlashBoot | free | already on; helps repeat starts, not the first pull |
| Weights baked in | free | already done; saves ~1.9 GB *after* the pull |
| `HVYM_PROXY_TIMEOUT=600` | free | rides it out instead of failing (default) |
| Warm the endpoint before a demo | ~$1.12/h while on | `scripts/warm.py on` — see [WARMING.md](WARMING.md) |
| Shrink the image | effort | the 3.96 GB torch layer is where the time goes |
| `workersMin=1` | **~$24/mo** | removes it entirely, gives up scale-to-zero |

`workersMin=1` costs roughly 40× the compute bill at 1,000 drawings/month, so it
buys latency, not economy — reach for it only if a demo genuinely cannot tolerate
the first call. `scripts/warm.py` wraps exactly that as an operator switch.

**Warming is designed differently for the demo than for the product**, because a
switch a client holds fails expensively while one an operator holds does not.
[WARMING.md](WARMING.md) covers both and why they diverge.

**A queued job is not a failed one.** `/runsync` caps at ~90 s server-side and
then returns the job still `IN_QUEUE`; the proxy polls `/status/{id}` from there.
Any client written against RunPod directly must do the same, or every
scale-from-zero request will look like a 502.

## Other modes of the same image

The entrypoint picks a role at run time, so one artifact covers every deployment:

```sh
docker run --gpus all -p 8000:8000 -e HVYM_API_KEY=... IMAGE serve   # persistent pod
docker run --gpus all IMAGE cli reangle --in a.png --out a.glb       # one-off
docker run --gpus all -it IMAGE bash                                  # debug
```

`serve` is option B from the design discussion: no proxy needed since the scoped
key is checked directly, but it bills continuously (~$250–540/mo) instead of
~$0.55/mo at 1,000 drawings.

## Operational checks

- `GET /healthz` on the proxy — open, reports `runpod_configured`, never echoes keys
- Worker logs show `warmed models: {...}` at startup; if a request instead pays the
  ~15 s load, warm-up is broken
- Worker logs show `front view: ... IoU=0.77` per request; a sharp drop means the
  mesh no longer matches the artist's silhouette
- `X-Cache: HIT` should dominate in normal use — one call per drawing
