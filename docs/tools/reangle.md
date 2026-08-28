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
| `mc_resolution` | int (64–512) | `256` | Marching-cubes grid. **TripoSR only** |
| `backbone` | str | `trellis` | `trellis` or `triposr` — see below |
| `texture_size` | int (256–4096) | `2048` | Baked texture resolution — see below |
| `target_faces` | int (2 000–200 000) | *backbone's own* | Unset: TRELLIS caps at 20 000, TripoSR uncapped |
| `seed` | int | `0` | TRELLIS only; fixed so results stay cacheable |

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

`matte → reconstruct → decimate → front-projected UV bake → glb`
(~1.78 s warm on TripoSR; TRELLIS reconstruction alone is 3.7–7.0 s):

1. **matte** (`core.imageio.isnet_matte`) — isnet → 512² RGBA, alpha = silhouette. A
   faithful port of `prep_input.py`; its exact normalisation is load-bearing.
2. **reconstruct** (`../../backbones/`) — matted RGBA → mesh. **Swappable backbone.**
   Each backbone owns its own input convention: TripoSR wants RGB on 0.5 grey,
   TRELLIS wants the alpha preserved so it skips its own rembg and builds from the
   same silhouette the UV bake aligns to.
3. **decimate** (`core.meshops`) — TRELLIS returns 176k–1.2M faces, capped at 20k.
   Before the bake, not after: UVs are per-vertex.
4. **uv-bake** (`uvbake.py`) — front-planar UV with the **original art** as texture.
5. **export** — embedded-texture `.glb`, so `ArmatureModel::load_from_memory` gets everything.

### The front axis is detected, never assumed

`detect_front_view()` scores all six axis-aligned views by silhouette IoU against the
artist's matte and picks the best (measured **0.776** vs 0.600 for the runner-up).

**The TRELLIS swap vindicated this.** TRELLIS's front axis is **+Y** (plane X,Z) where
TripoSR's is **+X** (plane Y,Z), and detection found it with no code change at all.

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
| `trellis` | single-image → 3D, **the default** | `backbones.trellis.load_trellis` |
| `triposr` | single-image → 3D, the rollback | `reconstruct.load_triposr` |

All three are *declared*, so either backbone is reachable wherever its weights
exist. Only `isnet` plus the **default** backbone are `models_needed()` and warmed
at startup — warming both would make each shipped image fail on weights it does not
carry. `warmup()` then runs one tiny reconstruction so CUDA/spconv kernels
initialise at startup rather than costing an artist ~57 s on the first real job;
it is deduplicated by model key, so sharing a worker with `mesh` costs one pass.

## Licensing — shippable

| Component | License | |
|---|---|---|
| TRELLIS (model + code) | **MIT** | ✅ |
| TripoSR (model + code) | **MIT** | ✅ |
| isnet (`isnet_dis.onnx`) | permissive | ✅ |
| trimesh, torchmcubes, onnxruntime | MIT / Apache-2.0 | ✅ |
| ~~Wonder3D / DrawingSpinUp~~ | **CC-BY-NC** | ❌ **not used** — demo only |

**No CC-BY-NC anywhere in this tool.** The prototype's Wonder3D dependency is deliberately
absent; if anyone reintroduces it as a backbone it must be flagged demo-only and must never
become the default (AGENTS.md §9).

## Known limits

- **A looser silhouette than TripoSR.** TRELLIS builds a fuller body than the drawn
  figure — IoU 0.675 vs 0.772, width/height 0.296 vs 0.271 — which shows as a smeared
  rim at the outline, not as misplaced art. It was accepted because TripoSR *tears*
  (ragged arms, a smeared ponytail) and a torn mesh is what an artist notices
  (BENCHMARK.md §6d). One untested lever could close much of it: fit the projection to
  the **matte's** bounding box rather than the mesh's.
- **Pose is the constraint, not medium.** Thin protrusions (outstretched arms, frilly
  skirts) collapse reconstruction; neutral stances reconstruct cleanly. Established for
  DrawingSpinUp — **not retested on either current backbone** (BENCHMARK.md §6),
  though TRELLIS's intact limbs on the test character are a point against it.
- **Small-angle ceiling (~±20°).** Beyond it the proxy's guessed sides and disocclusion
  holes show. Both backbones infer back surfaces well enough to widen this somewhat.
- **Back faces mirror-smear** under front-planar UV. Fix is xatlas unwrap + disocclusion
  inpainting (REANGLE_PIPELINE.md §9).
- **Never show the backbone's own texture** on the style path — it is the soft, style-lost
  version (§7.4).

## Environment

| Variable | Default | Notes |
|---|---|---|
| `ISNET_PATH` | `/workspace/models/isnet_dis.onnx` | |
| `TRIPOSR_PATH` | `/workspace/TripoSR` | repo must be importable |
| `TRIPOSR_CHUNK_SIZE` | `8192` | renderer chunking |
| `TRELLIS_PATH` | `/workspace/TRELLIS` | repo must be importable |
| `HVYM_REANGLE_BACKBONE` | `trellis` | operator override; the TripoSR image pins it |

Two pins are **not optional** (BENCHMARK.md §5): `transformers<5` (v5 renamed ViT weights and
the checkpoint will not load) and an `onnxruntime-gpu` matching the CUDA major version
(a mismatch falls back to CPU *silently* and costs 242× on the matte).
