# TRELLIS's own appearance, baked without a restricted rasteriser

**Question:** TRELLIS already decodes an appearance from the artist's drawing and
`reconstruct` throws it away (`formats=["mesh"]`). Is it worth having?

**Answer: not on the style path. Possibly for props — the test could not tell,
for a reason that is our fault and is fixable.**

Run 2026-08-28 on an isolated probe endpoint (`hvym-img-mesh:0.5.0-probe`,
created and destroyed for this; production was never touched). Subject is
[`../paint3d/source_drawing.png`](../paint3d/source_drawing.png), the same
drawing every other benchmark here uses.

## The images

| file | what it shows |
|---|---|
| **`style_zoom.png`** | **the decisive one.** Head and torso: the drawing, reangle's front projection, the gaussian bake. |
| `gaussian_turntable.png` | full 360°, 45° steps. 0° is the back, 180° the front. |
| `gaussian_atlas.png` | the two atlases. Left the xatlas 360° layout, right reangle's — which is just the drawing. |
| `gaussian_vs_projection.png` | both paths across ±40°, the window reangle is specified for. |
| `mesh_gaussian_probe.glb` | what the endpoint returned. 20k faces, 13,539 verts, 1024² texture, 1.33 MB. |

## What it settles

**The face does not survive.** `style_zoom.png` is the whole argument: the front
projection keeps the eyes, mouth, collar ruffle and shirt logo because those
*are* the artist's pixels; the gaussian bake has a blank mottled surface where
the face should be, a blocky ponytail, and the logo as a smear.

So [`uvbake.py`](../../../hvym_img_tools/tools/reangle/uvbake.py) §7.4 — never
put the backbone's predicted texture on the style path — is **confirmed, not
narrowed.** This was run to re-examine that rule and the rule won.

**The 360° coverage is real.** No holes, no black gutters, no seams; the body,
clothing and boots read correctly from every angle. For props and set dressing,
where nothing has a face, that is what `hallucinate` wanted — from the artist's
own drawing, with no second model, no extra weights, and no licence cost.

## What it does NOT settle, and why

The mush is at least partly the sampler's, not TRELLIS's. Measured:

| | head's share of atlas | texels for the head |
|---|---|---|
| gaussian bake (xatlas 360°, 1024²) | 5.36% | ~237 × 237 |
| reangle (front planar, 2048²) | 3.67% | ~392 × 392 |

237² is ample for a recognisable face, so the atlas was not the constraint.
**`ColorField` was.** At `res=256` over a figure whose long axis is 1.0 the voxel
edge is 0.0039, so a 0.12-tall head spans **31 voxel layers** — 237 texel rows
fed from 31 distinct values, a **7.7× under-resolution**. Matching them needs
`res ≈ 2048`, which is nearly free because the field is sparse: its cost scales
with the number of gaussians, not with res³.

**So this run does not measure TRELLIS's appearance fidelity.** It measures a
badly chosen default. The face verdict stands regardless — even a perfect bake
reconstructs the face rather than reproducing it — but "how good is TRELLIS's
texture really" is still open.

To answer it, one more round trip should **return the raw gaussian cloud**
alongside the glb, so every subsequent bake iteration is local and free instead
of costing a cold start.

## Cost

**22.8 s of work behind a 3,150 s queue.** The probe endpoint had no network
volume (deliberately — so it could not write into the cache production reads
from), so nothing was warm and the 6.5 GB image was pulled from scratch. Do not
read the 52 minutes as a property of the feature.

Gaussian decode + xatlas unwrap + bake costs ~18 s on top of the ~5 s mesh-only
path.
