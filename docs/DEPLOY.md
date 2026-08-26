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

## Installing on a server

One script, meant for a box that is already running something else:

```sh
curl -fsSL https://raw.githubusercontent.com/inviti8/hvym-img-tools/main/scripts/install_proxy.sh -o install_proxy.sh
less install_proxy.sh          # read it before running anything as root
sudo bash install_proxy.sh
```

It prompts for the RunPod key (hidden), the endpoint ID, and generates the
scoped client key. Re-running upgrades in place; `--status` and `--uninstall`
do what they say.

What it deliberately does **not** do is touch your web-server config. On a host
with another live service, a bad edit plus a reload takes that service down, so
the installer prints an nginx/Caddy snippet for you to paste instead. Re-print
it any time with `--print-proxy-conf`.

Other properties worth knowing:

- Secrets are written to `/etc/hvym-img-tools/proxy.env` (0600) and passed with
  `--env-file`, so they stay out of shell history and out of `ps`.
- It binds `127.0.0.1` only — nothing is exposed until you front it with TLS.
- It refuses to report success unless `/healthz` reports auth enabled **and** an
  unauthenticated request actually gets a 401. A misconfigured open proxy fails
  the install rather than quietly serving.
- Docker is detected, never installed: adding a container runtime to a box
  running production is a decision to make deliberately.


## Running the proxy

```sh
docker run -d --restart=unless-stopped -p 127.0.0.1:8080:8080   --memory=512m --memory-reservation=256m --cpus=0.5   -e HVYM_API_KEY="$(python -m hvym_img_tools.core.auth)"   -e RUNPOD_API_KEY=...      `# never leaves the server`   -e RUNPOD_ENDPOINT_ID=...    -e HVYM_MAX_UPLOAD_MB=8 -e HVYM_PROXY_TIMEOUT=600   ghcr.io/inviti8/hvym-img-proxy:0.1.0
```

### Measured footprint

`ghcr.io/inviti8/hvym-img-proxy:0.1.0`, real image, real uploads:

| State | Memory | CPU |
|---|---|---|
| Idle | **38-42 MiB** | 0.3 % |
| One real drawing (556 KB) | 44 MiB peak | - |
| **Six concurrent real drawings** | **74 MiB** | 1.8 % |
| One 8 MB upload | 104 MiB peak | - |
| One 30 MB upload (near the 32 MB default cap) | **280 MiB peak** | - |

**Peak scales at roughly 8x the upload size.** The proxy buffers the whole file,
base64-encodes it (1.33x), and the JSON serialisation and httpx both take copies.
That multiplier - not idle usage - is what sizing must survive.

RSS is *sticky* after a large upload (it sat at ~165 MiB before settling back to
~74 MiB), so budget against the high-water mark rather than the idle figure.

**`HVYM_MAX_UPLOAD_MB` is the load-bearing knob.** It defaults to 32, which
allows a 280 MiB spike per concurrent request. Character drawings are ~556 KB,
so **8 gives ~14x headroom and caps the spike near 105 MiB**. Lower it before
worrying about anything else.

## Co-hosting on a shared VPS

A 2 vCPU / 4 GiB box already running another service has ample room: expect
**~75 MiB steady** and well under 1 % of one core. The proxy is I/O-bound - it
spends nearly all its time awaiting RunPod, not computing.

Check what the neighbour actually leaves free first:

```sh
free -m; docker stats --no-stream
```

Give it a hard ceiling so it can never starve the neighbour. With `--memory=512m`
a runaway proxy is OOM-killed and restarted by `--restart=unless-stopped`, which
is the correct failure mode on a shared box - the other service keeps running.

Bind to `127.0.0.1:8080` (as above) so only the local reverse proxy can reach it.

### Reverse-proxy settings that will otherwise break it

If the box already terminates TLS for the other project, reuse that proxy rather
than adding a second. Two defaults will break this service:

- **`client_max_body_size` defaults to 1 MB in nginx** - every upload fails with
  413. Set it to match `HVYM_MAX_UPLOAD_MB`.
- **`proxy_read_timeout` defaults to 60 s in nginx** - this kills *every cold
  start*, which runs up to ~260 s (BENCHMARK.md 6b). Set 300 s or more.

```nginx
location /tools/ {
    proxy_pass http://127.0.0.1:8080;
    client_max_body_size 8m;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
}
```

```caddy
img.example.com {
    reverse_proxy 127.0.0.1:8080 {
        transport http { read_timeout 300s }
    }
    request_body { max_size 8MB }
}
```

Caddy gets a certificate automatically; with nginx use certbot. **TLS is not
optional** - over plain HTTP the scoped key is cleartext on the wire.


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
