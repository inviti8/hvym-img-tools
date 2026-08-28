"""`ReangleTool` — tool #1, and the reference implementation of the Tool contract.

Drawing in → textured `.glb` out: a rough 3D proxy carrying the artist's original
art front-projected as its texture. Inkternity loads it as a static model, orbits
the camera, and bakes the chosen view to canvas (REANGLE_PIPELINE.md §7).

Style is preserved by construction: we move the real pixels, never redraw them.

## Backbone

**TRELLIS is the default as of 0.3.0.** Measured on the character drawing, baked
through this pipeline (docs/BENCHMARK.md §6d):

- TRELLIS holds together where TripoSR tears. TripoSR's ponytail smears into a
  wedge at every angle and its arm boundaries come back ragged; TRELLIS returns
  coherent hair and intact limbs.
- It costs silhouette accuracy: IoU **0.675** against TripoSR's **0.772**, because
  TRELLIS builds a fuller body than the drawn figure (width/height 0.296 vs
  0.271). That shows up as a smeared rim, not as misplaced art.

Three ways back to TripoSR, cheapest first:

1. **Re-route the endpoint** — `set_tool_endpoint.sh --remove reangle` sends the
   tool back to the untouched TripoSR endpoint. No rebuild, no client change,
   and it is the only lever that also restores the wider GPU pool.
2. **`HVYM_REANGLE_BACKBONE=triposr`** on a worker that carries both.
3. **`backbone=triposr` per request**, likewise.

Levers 2 and 3 need TripoSR present in the serving image; lever 1 always works.
"""
from __future__ import annotations

import os

from pydantic import BaseModel, Field

from ...backbones import warm_kernels
from ...backbones.trellis import TRELLIS_MODEL_KEY, load_trellis
from ...core.imageio import DEFAULT_TEXTURE_SIZE
from ...core.tool import Context, FileBytes, MediaResponse, Tool
from . import reconstruct as backbones
from .pipeline import ISNET_MODEL_KEY, load_isnet, run_pipeline

#: Operator override, read per request so a restart is enough to change it.
BACKBONE_ENV = "HVYM_REANGLE_BACKBONE"

#: What ships when nothing says otherwise.
SHIPPING_BACKBONE = "trellis"


def default_backbone() -> str:
    """The backbone this worker uses when the request does not name one."""
    return os.environ.get(BACKBONE_ENV) or SHIPPING_BACKBONE


class ReangleInput(BaseModel):
    """Multipart: the rasterised selection plus a couple of knobs."""

    image: FileBytes = Field(description="Character drawing (PNG/JPEG), any size or background")
    mc_resolution: int = Field(
        default=backbones.MC_RESOLUTION_DEFAULT,
        ge=64,
        le=512,
        description=(
            "Marching-cubes grid. **TripoSR only** — TRELLIS decodes a structured "
            "latent rather than extracting an isosurface and ignores this. For "
            "TripoSR: 128 ~ 0.66s, 192 ~ 1.06s, 256 ~ 1.78s, 320 ~ 3.40s."
        ),
    )
    backbone: str = Field(
        default_factory=default_backbone,
        description=(
            "Reconstruction backbone: 'trellis' (default, sounder geometry) or "
            "'triposr' (tighter silhouette). Both MIT. A backbone this worker "
            "has no weights for is rejected rather than silently substituted."
        ),
    )
    texture_size: int = Field(
        default=DEFAULT_TEXTURE_SIZE,
        ge=256,
        le=4096,
        description=(
            "Baked texture resolution. 512 visibly softens the artist's linework once "
            "the mesh is magnified in-app; drawings are typically ~2K, so 2048 is at or "
            "below source rather than an upscale. Cheap -- linework compresses well. "
            "Note the silhouette edge does not sharpen past isnet's own 1024 input."
        ),
    )
    target_faces: int | None = Field(
        default=None,
        ge=2_000,
        le=200_000,
        description=(
            "Decimation target. Left unset it follows the backbone: TRELLIS caps at "
            "20,000 (176k-1.2M raw is not a shippable payload, and decimating to 20k "
            "cost 0.001 silhouette IoU), TripoSR is left uncapped at its ~30k."
        ),
    )
    seed: int = Field(
        default=0,
        ge=0,
        le=2**31 - 1,
        description=(
            "Fixed by default so a drawing is reproducible and cacheable. Only "
            "TRELLIS is stochastic; TripoSR ignores it. Change it to reroll."
        ),
    )


class ReangleTool(Tool):
    name = "reangle"
    summary = "Style-preserving camera reangle: drawing → textured .glb 3D proxy"
    # 0.2.0: texture defaults to 2048 (was 512).
    # 0.3.0: TRELLIS replaces TripoSR as the default backbone, and `target_faces`
    #        and `seed` join the request. The version is part of the cache key,
    #        so this is what stops a TripoSR result being served for a request
    #        that now means TRELLIS.
    version = "0.3.0"

    InputModel = ReangleInput
    OutputModel = MediaResponse  # binary .glb

    def model_loaders(self) -> dict[str, object]:
        """Every backbone this tool can serve, whether or not it is the default.

        Declaring both is what makes `backbone=triposr` a live rollback lever on
        an image that carries both sets of weights. `models_needed` keeps the
        one this worker does not default to lazy, so an image carrying only one
        of them still starts cleanly.
        """
        return {
            ISNET_MODEL_KEY: load_isnet,
            backbones.TRIPOSR_MODEL_KEY: backbones.load_triposr,
            TRELLIS_MODEL_KEY: load_trellis,
        }

    def models_needed(self) -> list[str]:
        """Warm the matte and the *default* backbone only.

        Warming both would make a reangle-only image fail on TRELLIS and a mesh
        image fail on TripoSR, for weights neither is going to be asked for.
        """
        return sorted({ISNET_MODEL_KEY, backbones.model_key_for(default_backbone())})

    def warmup(self, ctx: Context) -> None:
        """Initialise the backbone's CUDA kernels at startup, not on a request.

        TRELLIS made this mandatory: warm *weights* still left the mesh tool's
        first real job at ~57s against 4.1s for the next (docs/tools/mesh.md
        §6b). Deduplicated by model key, so sharing a worker with `mesh` costs
        one pass, not two.
        """
        name = default_backbone()
        key = backbones.model_key_for(name)
        warm_kernels(backbones.get_backbone(name, ctx.models.get(key)), key)

    def run(self, req: ReangleInput, ctx: Context) -> MediaResponse:
        # Resolve the model by the *requested* backbone rather than hard-coding
        # TripoSR's key. Passing a fixed key meant `backbone=trellis` handed a
        # TSR object to TrellisBackbone and failed deep inside someone else's
        # library -- the `backbone` field looked supported and was not.
        model_key = backbones.model_key_for(req.backbone)
        try:
            # Declared in model_loaders, so for the default backbone this is a
            # dict lookup rather than a ~14s load.
            model = ctx.models.get(model_key)
        except KeyError as exc:
            raise RuntimeError(
                f"backbone {req.backbone!r} needs model {model_key!r}, which this "
                f"worker does not carry. Either route the tool at an endpoint whose "
                f"image has it, or ask for one of {backbones.backbone_names()}."
            ) from exc

        backbone = backbones.get_backbone(req.backbone, model)
        # Unset means "whatever this backbone returns is right": TripoSR's ~30k
        # depth proxy passes through, TRELLIS's 176k-1.2M gets capped.
        target_faces = req.target_faces
        if target_faces is None:
            target_faces = backbones.default_target_faces_for(req.backbone)

        result = run_pipeline(
            req.image,
            isnet_session=ctx.models.get(ISNET_MODEL_KEY),
            backbone=backbone,
            mc_resolution=req.mc_resolution,
            texture_size=req.texture_size,
            target_faces=target_faces,
            seed=req.seed,
        )
        return MediaResponse(
            data=result.glb,
            media_type="model/gltf-binary",
            filename="char.glb",
        )
