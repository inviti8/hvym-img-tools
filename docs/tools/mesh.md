# mesh — sketch to an untextured 3D reference

**Status (2026-08-27): LIVE.** Endpoint `km99b7mrj2f85r`, image
`ghcr.io/inviti8/hvym-img-mesh:0.3.1`. Verified end to end through the proxy. Every number here is
measured; sources are in
[`../benchmark/paint3d/FINDINGS.md`](../benchmark/paint3d/FINDINGS.md).

**What it does.** One rough sketch in, one clean untextured `.glb` out — a 3D
reference the artist orbits and draws over, and keeps in a library to reuse
across frames and scenes.

```
rough sketch of a chair  ──▶  /tools/mesh  ──▶  untextured chair
                                                      │
                                    orbit it, draw over it, keep it
```

---

## 1. Why untextured is the whole point

This started as a generative *texturing* tool and arrived somewhere better. If
the mesh exists to be drawn over, its texture is going to be covered anyway —
so generating one is work the artist throws away.

Dropping it removes the only real objection in the earlier design:

| | textured (old plan) | **untextured (this)** |
|---|---|---|
| Backbone | TripoSR + Paint3D | **TRELLIS** |
| Licensing | SD 1.5 is **OpenRAIL-M** — first non-permissive thing we'd ship | **MIT end to end** |
| Work | ~25 s texture pass on top | **4–7 s total** |
| Weights in image | + 6–8 GB of SD/ControlNet/IP-Adapter | TRELLIS only |
| Failure mode | invents a character you didn't draw | none — it returns geometry, not art |

There is no style-drift risk here at all, because nothing is drawn. That is what
makes this a comfortable fit with the project's thesis in a way `hallucinate`
never quite was.

**This supersedes [`hallucinate.md`](hallucinate.md)** as the thing to build.
That document stays as the record of why generative texturing was rejected.

## 2. Contract

```
POST /tools/mesh        multipart/form-data
```

```python
class MeshInput(BaseModel):
    image: FileBytes = Field(description="Rough sketch (PNG/JPEG), any size or background")
    target_faces: int = Field(
        default=20_000, ge=2_000, le=200_000,
        description="Decimation target. 20k keeps ~99.5% of the silhouette at "
                    "~0.3 MB; raw TRELLIS output is 176k-1.2M faces and up to "
                    "20.8 MB, which is not shippable.",
    )
    seed: int = Field(
        default=0, ge=0, le=2**31 - 1,
        description="Fixed so results are reproducible and cacheable. Change to reroll.",
    )
```

Returns `model/gltf-binary` — **untextured**, same wire shape as `reangle`, so
Inkternity loads it through the identical `ArmatureModel::load_from_memory` path.

**Orientation is Z-up**, matching what `reangle` already returns. Worth stating
explicitly: TRELLIS emits Z-up and it silently cost two debugging rounds during
evaluation, once in a renderer and once in a Paint3D run.

## 3. Decimation is mandatory, and nearly free

Raw TRELLIS output is unusable as a payload. Measured, with proper triangle
rasterisation for the silhouette comparison:

| model | raw | at 20k faces | silhouette kept |
|---|---|---|---|
| chair | 331,560 faces / 5.7 MB | **0.3 MB** | **0.997** |
| rock | 1,214,456 faces / 20.8 MB | **0.3 MB** | **0.995** |
| character | 176,724 faces / 3.0 MB | **0.3 MB** | **0.994** |

The rock shrinks **69×** while keeping 99.5% of its outline. Even 10k faces holds
0.987–0.993.

`target_faces` is an absolute count, not a fraction, so the response size is
predictable regardless of what the model produced. At 20k a result is **smaller
than a current reangle `.glb`** (1.2 MB), which matters for a tool whose whole
point is accumulating a library.

## 4. Backbone: TRELLIS

| | |
|---|---|
| Code | **MIT** (microsoft/TRELLIS) |
| Weights | **MIT** (microsoft/TRELLIS-image-large, 2.6M downloads) |
| Work | 3.7–7.0 s + a one-off 14.2 s model load per worker |
| VRAM | **16 GB minimum** — constrains the GPU tier |

Chosen on evidence, not reputation. Against TripoSR on the same
largest-connected-component metric:

| subject | TripoSR | **TRELLIS** |
|---|---|---|
| chair | 57.7% (14 parts) | **98.6%** (4 parts) |
| rock | 93.4% (7 parts) | **100.0%** (2 parts) |
| character | — | 96% in the top two parts |

TripoSR cannot do this job: its implicit field does not connect thin structures,
and raising `mc_resolution` makes it *worse*, not better.

### Installation is the real cost

TRELLIS's own `setup.sh` fails silently on several submodules. Six fixes are
needed and the Dockerfile must encode all of them, with build-time assertions in
the style the GPU image already uses:

| problem | fix |
|---|---|
| `blinker` distutils conflict | `pip install --ignore-installed blinker` |
| relative `ckpts/...` read as a repo id → 401 | `snapshot_download`, then `chdir` before `from_pretrained` |
| `xformers` missing | `xformers==0.0.28.post1` from the cu124 index |
| `xops.fmha.BlockDiagonalMask` moved | alias from `xops.fmha.attn_bias` |
| `spconv` missing | `spconv-cu124`, `utils3d` |
| kaolin numpy ABI error | `numpy<2` |

`nvdiffrast` is **not** required — it is only used for rendering.

Assert at build time that the pipeline imports and produces a mesh, or a silently
broken image will reach a worker.

## 5. Deployment

**Its own image and endpoint, initially.** TRELLIS plus its compiled submodules
is a large addition, and folding it into the reangle worker would lengthen an
already ~260 s cold start for every reangle request.

The **16 GB VRAM floor** narrows the GPU tier: the current endpoint's 4090/L4
(both 24 GB) qualify, but there is no headroom to drop to a cheaper 16 GB card.

The proxy needs a tool→endpoint map instead of a single `RUNPOD_ENDPOINT_ID` —
the same change `hallucinate.md` identified.

**Convergence happened.** As of reangle 0.3.0 TRELLIS is *also* reangle's default
backbone, so this image serves both tools and one warm TRELLIS pipeline serves
both (`backbones.warm_kernels` dedupes the startup pass by model key). The
evidence for the swap is docs/BENCHMARK.md §6d; the image gained isnet and
`onnxruntime-gpu` for reangle's matte and nothing else.

The reangle image and endpoint stay live and untouched as the rollback target —
`set_tool_endpoint.sh --remove reangle` is the whole procedure.

## 6. Supporting the library

The result cache is keyed by `sha256(image + params)`, which is exactly a content
address — **the same sketch always yields the same mesh, instantly**. That makes
it a natural asset id for Inkternity's library.

Return it to the client so the library can key on it:

```
X-Cache-Key: <sha256>
```

Two consequences worth designing for now rather than retrofitting:

- **Re-import is free.** A sketch already meshed returns from cache in ~0.05 s,
  so the client need not store the `.glb` itself if it prefers to re-fetch.
- **Reuse is where the value is.** A reference drawn over once is a novelty; a
  reference placed in twenty frames changes how a scene gets built. The 4–7 s
  cost amortises to nothing.

## 6b. Measured on the live endpoint

Chair and rock sketches through the real chain (proxy → RunPod → worker):

| | |
|---|---|
| Steady-state work | **~5 s** |
| First job on a fresh worker | **5.4 s** (was 57 s before the warm-up fix) |
| Cache hit | 0.015 s |
| Output | exactly 20,000 faces, **0.35 MB**, untextured, 99.9% one component |
| `X-Cache-Key` | returned, stable across repeats |

Decimation and file size landed exactly where the design predicted, and the
delivered chair still reads as a chair (`../benchmark/mesh_endpoint_live.png`).

### Kernel warm-up — found here, fixed in 0.3.1

**On 0.3.0 a held lease reported `warm` and the next real job still cost ~57 s**,
with only the job after it dropping to 4.1 s. A 14× swing a client would
experience as the tool being wildly inconsistent.

The cause is our own keepalive. `serverless.py` short-circuits `__warm__` before
any pipeline work, precisely so a ping every few seconds costs no GPU time. That
keeps the *process* alive but never runs a kernel, so CUDA/spconv initialisation
is still paid by the first genuine request. For TripoSR this was invisible —
its whole job is ~2 s. TRELLIS makes it a ~54 s surprise.

**Fixed in 0.3.1** by `Tool.warmup()`, called once from `init()` after models
are warmed: the mesh tool runs one tiny reconstruction so the kernels initialise
at worker startup rather than on an artist's first request. Deliberately *not*
in the `__warm__` ping, which fires every few seconds and must stay free.

Measured after the fix, on a fresh worker:

| | 0.3.0 | **0.3.1** |
|---|---|---|
| First job | 57.0 s | **5.4 s** |
| Second job | 4.1 s | 5.5 s |

First and second are now the same speed — the cliff is gone.

## 7. What is measured, and what is not

**Measured:** reconstruction quality on a chair, a rock and a character;
decimation cost; file sizes; timings; licences; the full install path.

**Not measured, and worth knowing before shipping:**

1. **Peak VRAM.** Only the 16 GB documented floor is known — the actual peak on
   our inputs was never recorded, and it decides whether a 24 GB tier is
   comfortable or marginal.
2. **Cold start for this image.** TRELLIS weights plus submodules will differ
   from reangle's measured ~260 s.
3. **How it handles a genuinely bad sketch.** All three test inputs were clean.
   A scribble may produce a confident, wrong mesh — and there is no equivalent of
   reangle's silhouette IoU to detect that.
4. **Multi-view.** TRELLIS has `run_multi_image()`, which would let an artist
   reangle → draw the next angle → feed both back. That is a stronger answer to
   disocclusion than any generative fill, because every view stays the artist's
   own linework. Untested, and the most interesting thing on this list.

## 8. Naming

`mesh` describes what it returns. It is deliberately not `hallucinate`, which was
honest when the tool invented texture and would be misleading now — the geometry
follows the artist's sketch rather than reinterpreting it.
