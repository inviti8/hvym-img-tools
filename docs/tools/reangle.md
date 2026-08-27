# reangle — style-preserving camera-angle adjustment

**Tool #1**, and the reference implementation of the `Tool` contract.

Drawing in → **textured `.glb`** out: a rough 3D proxy of the character carrying the
artist's **original art front-projected** as its texture. Inkternity loads it as a static
model, orbits the camera, and bakes the chosen view to canvas.

Style is preserved *by construction* — we move the artist's real pixels, we never redraw
them. The authoritative pipeline spec is
[`../../../infinipaint/docs/design/REANGLE_PIPELINE.md`](../../../infinipaint/docs/design/REANGLE_PIPELINE.md);
measured performance is in [`../BENCHMARK.md`](../BENCHMARK.md).

## Endpoint

```
POST /tools/reangle        multipart/form-data
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `image` | file | *required* | Character drawing, any size or background |
| `mc_resolution` | int (64–512) | `256` | Marching-cubes grid — the main cost lever |
| `backbone` | str | `triposr` | Reconstruction backbone |
| `texture_size` | int (256–4096) | `2048` | Baked texture resolution — see below |

Returns `model/gltf-binary` (`char.glb`). Responses carry `X-Cache: HIT|MISS` and
`X-Tool-Version`. Results are cached by `sha256(image + params)`, so re-requesting the same
drawing is instant — which is the whole access pattern: **one call per drawing**, then all
interaction is local in Inkternity.

Integrating a client? [CLIENT.md](../CLIENT.md) is the full contract — status
codes, retry semantics, and the ≥300 s timeout a cold start requires.

```sh
curl -X POST http://localhost:8000/tools/reangle \
     -F image=@drawing.png -F mc_resolution=256 -o char.glb

uv run hvym-img reangle --in drawing.png --out char.glb   # same thing, locally
```

### Texture resolution

Defaults to **2048** (it was 512). At 512 the artist's linework visibly softened
once the mesh was magnified in-app — individual hair strands and eye detail
turned to mush. Source drawings are typically ~2K, so 2048 is at or below the
original rather than an upscale.

It is cheap: linework compresses well, measuring 342–634 KB of PNG at 2048
against a ~680 KB mesh that already dominates the `.glb`.

Two things worth knowing:

- **The silhouette edge does not sharpen past ~1024.** The alpha comes from
  isnet, whose own input is 1024², so raising this recovers *interior* linework
  from the source image, not a crisper outline.
- **Alignment is unaffected.** `fit_to_frame`'s output is divided by `size - 1`
  to give normalised UVs, so the frame size cancels — verified identical to
  2.2e-16 across 512–4096. Only `DEFAULT_MARGIN` moves the projection, and it is
  unchanged, so BENCHMARK.md §2's 0.776 silhouette IoU still holds.

## Pipeline

`matte → reconstruct → front-projected UV bake → glb` (~1.78 s warm, RTX 4090):

1. **matte** (`core.imageio.isnet_matte`) — isnet → 512² RGBA, alpha = silhouette. A
   faithful port of `prep_input.py`; its exact normalisation is load-bearing.
2. **reconstruct** (`reconstruct.py`) — single image → mesh. **Swappable backbone.**
3. **uv-bake** (`uvbake.py`) — front-planar UV with the **original art** as texture.
4. **export** — embedded-texture `.glb`, so `ArmatureModel::load_from_memory` gets everything.

### The front axis is detected, never assumed

`detect_front_view()` scores all six axis-aligned views by silhouette IoU against the
artist's matte and picks the best (measured **0.776** vs 0.600 for the runner-up).

This exists because assuming the conventional `(X, Y)` image plane **silently projects the
art onto the mesh from the side** — TripoSR's front axis is **+X**, so the plane is `(Y, Z)`.
That bug shipped in the first benchmark bake and is now covered by regression tests. Since
the backbone is swappable, detecting beats hard-coding: a different backbone brings a
different convention.

Coarse meshes are densified by face sampling before scoring — a low-poly mesh has too few
vertices for a splat to approximate a silhouette, and every axis would score ~0.

## Models

| Key | What | Loader |
|---|---|---|
| `isnet` | matting / background removal | `pipeline.load_isnet` |
| `triposr` | single-image → 3D | `reconstruct.load_triposr` |

Both are declared in `models_needed()` and warmed at startup by `ModelCache`, so no request
pays the ~13.7 s load. Paths come from `ISNET_PATH` / `TRIPOSR_PATH`.

## Licensing — shippable

| Component | License | |
|---|---|---|
| TripoSR (model + code) | **MIT** | ✅ |
| isnet (`isnet_dis.onnx`) | permissive | ✅ |
| trimesh, torchmcubes, onnxruntime | MIT / Apache-2.0 | ✅ |
| ~~Wonder3D / DrawingSpinUp~~ | **CC-BY-NC** | ❌ **not used** — demo only |

**No CC-BY-NC anywhere in this tool.** The prototype's Wonder3D dependency is deliberately
absent; if anyone reintroduces it as a backbone it must be flagged demo-only and must never
become the default (AGENTS.md §9).

## Known limits

- **Pose is the constraint, not medium.** Thin protrusions (outstretched arms, frilly
  skirts) collapse reconstruction; neutral stances reconstruct cleanly. Established for
  DrawingSpinUp — **not yet retested on TripoSR** (BENCHMARK.md §6).
- **Small-angle ceiling (~±20°).** Beyond it the proxy's guessed sides and disocclusion
  holes show. TripoSR's better back-surface inference widens this somewhat.
- **Back faces mirror-smear** under front-planar UV. Fix is xatlas unwrap + disocclusion
  inpainting (REANGLE_PIPELINE.md §9).
- **512² texture.** The warp is resolution-independent; a higher-res atlas is a
  straightforward quality win.
- **Never show the backbone's own texture** on the style path — it is the soft, style-lost
  version (§7.4).

## Environment

| Variable | Default | Notes |
|---|---|---|
| `ISNET_PATH` | `/workspace/models/isnet_dis.onnx` | |
| `TRIPOSR_PATH` | `/workspace/TripoSR` | repo must be importable |
| `TRIPOSR_CHUNK_SIZE` | `8192` | renderer chunking |

Two pins are **not optional** (BENCHMARK.md §5): `transformers<5` (v5 renamed ViT weights and
the checkpoint will not load) and an `onnxruntime-gpu` matching the CUDA major version
(a mismatch falls back to CPU *silently* and costs 242× on the matte).
