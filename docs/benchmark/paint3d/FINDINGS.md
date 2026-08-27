# Paint3D / MVPaint evaluation — 2026-08-27

Ran against the live reangle mesh on a rented RTX 4090 (~1 h, ~$0.74). The
question was whether a generative texture model improves on our front-projection,
particularly for regions the projection cannot cover.

**Verdict: no, not as a texture source. Keep the projection.** The one idea worth
keeping is narrower than it first looked — see [What survives](#what-survives).

---

## MVPaint — ruled out before testing

| | |
|---|---|
| License | **NONE** — no LICENSE file in the repo |
| Weights | a Dropbox link, no stated terms |
| Last push | 2025-08-22 |

Absent a license, all rights are reserved. That is a harder blocker than
Wonder3D's CC-BY-NC, which at least states its terms. Not tested further.

## Paint3D — tested end to end

| | |
|---|---|
| Code | Apache-2.0, real LICENSE file |
| UV-position ControlNet weights | Apache-2.0 |
| Base SD 1.5 | CreativeML OpenRAIL-M — commercial use allowed, not OSI-permissive |
| Renderer | kaolin 0.17.0 — installs cleanly on sm_89 with torch 2.4.1+cu124 |

Legally usable. The problem is what it produces.

### What was run

1. **`pipeline_UV_only.py`** — texture from UV-position control alone. Worked;
   **5.0 s** of diffusion.
2. **`pipeline_paint3d_stage1.py`** — multi-view depth-conditioned generation,
   IP-Adapter conditioned on `alice_char2.png`, on **our TripoSR mesh**.
   **24.8 s** processing (2m20s wall including model loads).
3. **`pipeline_paint3d_stage2.py`** — UV inpainting + tile refinement on stage 1's
   atlas. **10.3 s + 11.3 s** (4m17s wall).

Setup notes for anyone repeating this: the pinned cu113 stack is unnecessary —
a modern torch works, which also frees you from sm_86. Three fixes were needed:
`--ignore-installed blinker`, `huggingface_hub==0.20.3` (0.26 removed
`cached_download`, which `diffusers==0.25.0` still imports), and `numpy<2`
(`trimesh 3.20.2` calls the removed `np.product`).

### Finding 1 — it regenerates the character, it does not move it

`paint3d_generated_view.png` next to `source_drawing.png` is the whole answer.
Even with IP-Adapter conditioned on the artist's own drawing, the output is a
*different character*: different face, different proportions, Stable Diffusion's
house style. Compare `ours_projected_atlas.png`, which is the artist's exact
pixels because it is literally their drawing resampled.

This is precisely the failure REANGLE_PIPELINE.md §1 rejects generative reangle
for. Image conditioning narrows the drift; it does not remove it.

### Finding 2 — UV inpainting fails on a marching-cubes mesh

`paint3d_stage1_atlas.png` shows our TripoSR mesh xatlas-unwrapped: hundreds of
tiny disconnected islands (magenta = unpainted). TripoSR's marching-cubes output
has no clean charts to unwrap.

`paint3d_stage2_uv_inpainted.png` is what the UV inpainter did with it:
**formless grey mush across the entire atlas**, destroying even the regions that
had content. A 2D diffusion model operating in UV space assumes neighbouring
texels are spatially related. In a shattered atlas they are not, so it has
nothing coherent to reason about.

This matters beyond Paint3D: **any UV-space inpainter will struggle with this
mesh topology.** The obstacle is our geometry, not their model.

### Finding 3 — the assumed prerequisite was wrong, but for a different reason

Paint3D calls `xatlas` itself, so it does not need us to unwrap first — the
prerequisite identified before testing does not exist. But Finding 2 shows that
automatic unwrap of a marching-cubes mesh is exactly what breaks the inpainter.
Being handed the unwrap for free is no help when the unwrap is the problem.

---

## What survives

Two things are worth keeping from this:

**Lighting-less output is the right target.** Paint3D generates albedo with no
baked shading, and that instinct is correct for flat 2D linework — a texture
carrying its own highlights would fight the art. Our projection already has this
property for free, since the artist's drawing *is* the albedo.

**Disocclusion is still unsolved, and still bounded.** `front_planar_uv` projects
every vertex onto the front plane, so back faces mirror-smear the front art —
there are no holes to fill today, only smear. REANGLE_PIPELINE.md §9 caps the
feature at ~±20°, which keeps the smear off-screen. If that ceiling ever needs
raising, the cheap fixes come first: forward-warp with push-pull / edge-extend
fill, measured in milliseconds, no model, no license question, no style risk.

## Cost, for the record

Paint3D stage 1 + 2 is **~56 s of GPU work** against the current pipeline's
**1.9 s total**. Roughly 30× the compute to produce a texture that is worse for
this use case. Even had the quality held up, it would have rewritten the cost
model in BENCHMARK.md §2.

## Files

| | |
|---|---|
| `comparison.png` | the three-panel summary — start here |
| `source_drawing.png` | the artist's input |
| `ours_projected_atlas.png` | our front projection (the artist's pixels) |
| `paint3d_generated_view.png` | Paint3D's generated view — the style drift |
| `paint3d_stage1_atlas.png` | the fragmented xatlas unwrap (magenta = unpainted) |
| `paint3d_inpaint_mask.png` | what stage 2 tried to fill |
| `paint3d_stage2_uv_inpainted.png` | the grey mush it produced |
| `paint3d_stage2_uv_tiled.png` | after tile refinement — no better |
| `paint3d_uvonly_texture.png` | UV-position control alone, no image conditioning |
