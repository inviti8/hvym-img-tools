# TRELLIS's own appearance, baked without a restricted rasteriser

**Question:** TRELLIS already decodes an appearance from the artist's drawing and
`reconstruct` throws it away (`formats=["mesh"]`). Is it worth having?

**Answer: not for reangle. Yes, plausibly, for props.**

Run 2026-08-28 on an isolated probe endpoint (`hvym-img-mesh:0.5.x-probe`,
created and destroyed for this; production was never touched). Subject is
[`../paint3d/source_drawing.png`](../paint3d/source_drawing.png), the same
drawing every other benchmark here uses.

## Read this first: the first probe's diagnosis was wrong

The first run came back with **no face at all**, and this file originally
recorded that as a resolution problem — the colour field at `res=256` resolving
31 layers across a head the atlas gave 237 texel rows. That reasoning was
plausible, quantified, and **wrong**.

Fetching the raw cloud made the bake free to re-run, and the sweep settled it:

| what was varied | effect on the face |
|---|---|
| field resolution, 256 → 4096 | **none** |
| texture size, 1024 → 4096 | **none** |
| surface samples, 2M → 8M | **none** |
| **voxel shape, anisotropic → cubic** | **the face comes back** |

`anisotropy_isolated.png` is the isolation: same resolution, same texture, same
sample count, only the voxel shape differs. Left is a featureless blob, right
has eyes, brows, nose and mouth.

**The bug was voxel anisotropy.** `ColorField` normalised each axis
independently, so a figure measuring 1.0 × 0.293 × 0.227 got voxels **3.4×
taller than they were wide**. Horizontal features — eyes, mouth, the line of a
collar — were averaged across vertically while the horizontal axis was
*over*-sampled. Every reangle subject is a standing figure with roughly those
proportions, so this would have hit all of them.

A caution for anyone extending this: **the sharpness metric scored the broken
bake higher than the fixed one** (39.0 against 35.8). Laplacian variance rewards
the blocky noise anisotropy produces. `resolution_sweep.png` carries those
numbers and they are worth nothing; the images decided this, not the metric.

## What the fixed bake looks like

`style_zoom_fixed.png` — the drawing, reangle's projection, the first probe, and
the fixed bake, side by side.

The face is **recognisably the character**, and much softer than the drawing.
That softness is now understood and is not fixable by tuning:

| | |
|---|---|
| gaussians in the cloud | 133,792 |
| on the head | 17,946 (13.4%) |
| head gaussian spacing | 0.00217 |
| **gaussians across the head, top to bottom** | **~54** |
| texel rows the atlas gives that head | 237 at 1024px, 474 at 2048px |

**~54 colour samples have to fill 237+ texel rows.** That is TRELLIS's own
output density, not a sampler artefact, and it is why field resolution above
~460 does nothing: past that, each gaussian simply gets its own voxel and no new
information enters.

## Verdicts

**reangle: unchanged.**
[`uvbake.py`](../../../hvym_img_tools/tools/reangle/uvbake.py) §7.4 — never put
the backbone's predicted texture on the style path — **stands.** The projection
reproduces the artist's exact pixels; the bake reconstructs them from ~54
samples across a face. Even a perfect bake would be a reconstruction, so no
amount of sampler work changes this verdict.

**Props: promising, and the reason to keep this code.**
`gaussian_turntable_fixed.png` shows full 360° coverage with no holes, no
seams, and clothing and boots that read correctly from every angle. Nothing in
the props case has a face, so the one measured weakness does not apply — and it
costs no second model, no extra weights, and no licence compromise, which is
exactly what [`../../tools/hallucinate.md`](../../tools/hallucinate.md) §4 and §6
were struggling with.

## Files

| file | what it shows |
|---|---|
| `anisotropy_isolated.png` | **the decisive one.** Anisotropic vs cubic voxels, everything else held fixed. |
| `style_zoom_fixed.png` | drawing / projection / first probe / fixed bake. |
| `gaussian_turntable_fixed.png` | 360°, 45° steps, cubic bake. 180° is the front. |
| `resolution_sweep.png` | field resolution 256–4096. Flat, which is the point. |
| `style_zoom.png`, `gaussian_turntable.png`, `gaussian_vs_projection.png`, `gaussian_atlas.png` | the first probe, kept as the record of the wrong answer. |
| `appearance.npz` | the raw cloud + mesh, 2.4 MB. **Re-bake offline from this; do not spend another cold start.** |
| `mesh_gaussian_fixed.glb` / `mesh_gaussian_probe.glb` | fixed and original bakes. |

## Cost, and the lesson about it

| run | queue | work |
|---|---|---|
| first probe (`texture="gaussian"`) | 3,150 s | 22.8 s |
| cloud fetch (`texture="cloud"`) | 294 s | 6.1 s |

The first run bought exactly one answer about one configuration, and that answer
was wrong. The second brought back the cloud, after which resolution, texture
size, sample count and voxel shape were all tested locally for free — and the
real cause turned up in the variable nobody had suspected.

**Fetch the cloud first.** `texture="cloud"` exists for this.

Incidental confirmations from the cloud, both previously assumed: the SH
convention is right (`sh_in_range` 0.9999 against 0.589 read raw), and the mesh
and gaussian share a coordinate frame (extents `[0.293, 0.23, 1.0]` and
`[0.288, 0.227, 0.996]`).
