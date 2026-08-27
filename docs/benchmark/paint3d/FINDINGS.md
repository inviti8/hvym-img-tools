# Paint3D / MVPaint evaluation — 2026-08-27

Ran against the live reangle mesh on a rented RTX 4090 (~1 h, ~$0.74). The
question was whether a generative texture model improves on our front-projection,
particularly for regions the projection cannot cover.

**Verdict: no, not as a texture source. Keep the projection.** A follow-up test
of the seeded low-denoise hybrid produced one genuinely reusable result — the
style-preservation lever is compositing, not `denoising_strength` — but also
confirmed that our mesh topology blocks the whole UV-space approach.

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

## Follow-up: the seeded low-denoise experiment

The idea worth testing was a hybrid — seed the model with the artist's own
pixels instead of noise, then run at low `denoising_strength` so it only extends
rather than reinvents. Paint3D's UV models expect the mesh's xatlas layout, so
the drawing was re-baked into that atlas via Paint3D's own UV-position render:
front-facing texels get the artist's pixels, the rest are left genuinely empty.
That produced a **52.2% seeded / 23.4% holes** atlas — real disocclusion holes,
which `front_planar_uv` never creates because it mirror-smears instead.

Then a sweep: `denoising_strength` ∈ {0.30, 0.45, 0.60, 0.75, 1.00} ×
`guidance_scale` ∈ {3, 7}, fixed seed, IP-Adapter on the artist's drawing.
"Drift" is mean |Δ| on texels the artist actually painted — style override,
ideally zero.

### The dial does nothing

| denoise | raw drift | composited drift | hole fill |
|---|---|---|---|
| 0.30 | 6.37 | **0.00** | 71.9 % |
| 0.45 | 6.36 | **0.00** | 73.1 % |
| 0.60 | 6.35 | **0.00** | 74.6 % |
| 0.75 | 6.36 | **0.00** | 74.7 % |
| 1.00 | 6.38 | **0.00** | 74.7 % |

Across the entire sweep, raw drift moves by **0.29/255 — 0.1%**. Turning
`denoising_strength` from 0.3 to 1.0 does not meaningfully protect the artist's
pixels.

The reason is that the constant ~6.4 loss is not the diffusion at all: the
inpainting pipeline VAE-encodes and decodes the *whole* 1024² atlas, so the
unmasked region takes a round-trip hit no dial can tune away.

### The right lever is compositing, not the dial

Keep the artist's pixels verbatim outside the mask and take the model's output
only inside it. Drift becomes **exactly 0.00** — perfect preservation by
construction rather than by hoping a parameter is low enough.

That inverts the tuning intuition. "Minimal style injection" is not achieved by
turning the model down; it is achieved by **masking it out where the art exists,
and then turning it up where it is allowed to work** — higher denoise gives
better hole fill (74.7% vs 71.9%) and costs nothing, because inside a hole there
is no artist's work to protect.

This is a reusable result: it applies to any generative fill we ever bolt onto
the projection, not just Paint3D.

### But on this mesh it is academic

The atlas our TripoSR mesh unwraps into:

| | |
|---|---|
| UV islands | **3,035** |
| Median island | **2 texels** (~1.4 px across) |
| Islands under 100 texels | 2,679 — **88%** |
| Largest island | 3.2% of the surface |

A median island of two texels cannot carry linework, and neither the artist's
art nor a diffusion model can do anything useful with slivers that size. That is
why `seeded_comparison.png` is nearly featureless grey in all three panels — the
detail is lost in the *bake*, before any model runs.

So the compositing insight is sound and worth keeping, but it cannot be
exercised until the mesh unwraps into coherent charts. **Mesh topology is the
blocker, not the texture model** — the same conclusion Finding 2 reached from
the other direction.

## Follow-up 2: does remeshing give coherent charts?

Yes — with a real trade-off, and a different sweet spot per tool. Run entirely
on CPU (decimation and xatlas need no GPU), measuring UV islands after unwrap
and silhouette IoU against the original mesh.

| method | faces | UV islands | vs baseline | silhouette kept |
|---|---|---|---|---|
| baseline (TripoSR) | 29,887 | 1,015 | — | 1.000 |
| decimate 25% | 7,471 | 248 | 4.1× | **0.982** |
| decimate 10% | 2,988 | 173 | 5.9× | 0.976 |
| decimate 5% | 1,494 | 120 | 8.5× | 0.959 |
| voxel remesh 0.015 | 7,554 | 104 | 9.8× | 0.855 |
| voxel remesh 0.02 | 4,460 | 74 | 13.7× | 0.809 |
| voxel remesh 0.03 | 1,996 | **20** | **50.8×** | 0.772 |

**Decimation is close to free.** Quadric decimation to 10% of the faces gives
~6× fewer charts while keeping 97.6% of the silhouette. For reangle — where the
proxy matching the artist's outline is the whole point — this is a clear win with
no meaningful downside, and it also cuts mesh size ~10×.

**Voxel remesh trades silhouette for coherence.** It reaches 20 charts with a
median island of ~15,900 texels, which is a genuinely usable atlas, but at 0.772
IoU.

### The 0.772 is not what it looks like, and the obvious fix does not work

Rendering the silhouettes (`remesh_silhouettes.png`, blue = added, red = lost)
shows the voxel results are almost entirely **blue**: at pitch 0.02 it loses only
**0.1%** of the original silhouette while *adding* 23.5%. It inflates rather than
erodes — essentially no detail is destroyed.

That looked correctable, so it was tested. It is not:

| correction | result |
|---|---|
| erode one voxel cell before marching cubes | overshoots badly — 21.8% of the silhouette lost |
| shrink along vertex normals by 0.35–0.7 × pitch | 23.5% → 23.0% added. No real effect. |

The reason is that the inflation is not a surface offset. `.fill()` at a coarse
pitch **bridges gaps** — between the legs, between arm and torso — which is
topological merging that no offset can undo. It is also exactly *why* the chart
count collapses: it welds separate surfaces into one blob. The chart win and the
silhouette cost are the same mechanism, so they cannot be separated this way.

### Operating points

- **reangle** — decimate to ~10%. 5.9× fewer charts, 97.6% silhouette kept, and
  a 10× smaller mesh for free. Worth adopting on its own merits.
- **hallucinate** — voxel remesh is viable where silhouette fidelity is secondary
  to a clean atlas. Note that `alice_char2` is close to the worst case for it:
  thin limbs, a ponytail, a gap between the legs. **A chair or a rock has far
  fewer gaps to bridge**, so the cost should be substantially lower on the props
  this tool is actually aimed at — untested, and the obvious next measurement.

Either way the topology blocker is real but not fundamental: **a 6× to 50×
reduction in chart count is available, and the cheap end of that range is
free.** UV-space work is no longer ruled out.

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
