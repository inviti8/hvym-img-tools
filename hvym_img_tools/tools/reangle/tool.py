"""`ReangleTool` — tool #1, and the reference implementation of the Tool contract.

Drawing in → textured `.glb` out: a rough 3D proxy carrying the artist's original
art front-projected as its texture. Inkternity loads it as a static model, orbits
the camera, and bakes the chosen view to canvas (REANGLE_PIPELINE.md §7).

Style is preserved by construction: we move the real pixels, never redraw them.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ...core.imageio import DEFAULT_TEXTURE_SIZE
from ...core.tool import Context, FileBytes, MediaResponse, Tool
from . import reconstruct as backbones
from .pipeline import ISNET_MODEL_KEY, load_isnet, run_pipeline


class ReangleInput(BaseModel):
    """Multipart: the rasterised selection plus a couple of knobs."""

    image: FileBytes = Field(description="Character drawing (PNG/JPEG), any size or background")
    mc_resolution: int = Field(
        default=backbones.MC_RESOLUTION_DEFAULT,
        ge=64,
        le=512,
        description=(
            "Marching-cubes grid. Cost scales ~res³ and this is ~74% of wall-clock: "
            "128 ≈ 0.66s, 192 ≈ 1.06s, 256 ≈ 1.78s, 320 ≈ 3.40s."
        ),
    )
    backbone: str = Field(
        default="triposr",
        description="Reconstruction backbone. TripoSR (MIT) is the shippable default.",
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


class ReangleTool(Tool):
    name = "reangle"
    summary = "Style-preserving camera reangle: drawing → textured .glb 3D proxy"
    # 0.2.0: texture defaults to 2048 (was 512). Bumped because the version is
    # part of the cache key -- old 512 results must not be served for new requests.
    version = "0.2.0"

    InputModel = ReangleInput
    OutputModel = MediaResponse  # binary .glb

    def model_loaders(self) -> dict[str, object]:
        return {
            ISNET_MODEL_KEY: load_isnet,
            backbones.TRIPOSR_MODEL_KEY: backbones.load_triposr,
        }

    def run(self, req: ReangleInput, ctx: Context) -> MediaResponse:
        # Loaders were registered when the app was built, so these are already
        # warm and `get()` is a dict lookup rather than a ~13.7s load.
        backbone = backbones.get_backbone(
            req.backbone, ctx.models.get(backbones.TRIPOSR_MODEL_KEY)
        )
        result = run_pipeline(
            req.image,
            isnet_session=ctx.models.get(ISNET_MODEL_KEY),
            backbone=backbone,
            mc_resolution=req.mc_resolution,
            texture_size=req.texture_size,
        )
        return MediaResponse(
            data=result.glb,
            media_type="model/gltf-binary",
            filename="char.glb",
        )
