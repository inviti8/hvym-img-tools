"""`MeshTool` — sketch to an untextured 3D reference (docs/tools/mesh.md).

The counterpart to `reangle`, not a variant of it. Reangle moves the artist's
own pixels onto a proxy; this returns bare geometry for them to orbit and draw
over, and to keep in a library and reuse across frames.

Untextured on purpose: the surface gets drawn over, so generating one is work
the artist throws away — and dropping it keeps the tool MIT end to end, where
generative texturing would have brought Stable Diffusion's OpenRAIL-M licence
with it (docs/tools/hallucinate.md records that decision).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ...backbones import get_backbone, warm_kernels
from ...backbones.trellis import TRELLIS_MODEL_KEY, load_trellis
from ...core.tool import Context, FileBytes, MediaResponse, Tool
from .pipeline import TARGET_FACES_DEFAULT, run_pipeline


class MeshInput(BaseModel):
    """Multipart: the sketch, plus how much geometry to keep."""

    image: FileBytes = Field(description="Rough sketch (PNG/JPEG), any size or background")
    target_faces: int = Field(
        default=TARGET_FACES_DEFAULT,
        ge=2_000,
        le=200_000,
        description=(
            "Decimation target, an absolute count so the response size is "
            "predictable whatever the model produced. 20k keeps ~99.5% of the "
            "silhouette at ~0.3 MB; raw output is 176k-1.2M faces, up to 20.8 MB."
        ),
    )
    seed: int = Field(
        default=0,
        ge=0,
        le=2**31 - 1,
        description=(
            "Fixed by default so a sketch is reproducible and cacheable. Change "
            "it to reroll."
        ),
    )


class MeshTool(Tool):
    name = "mesh"
    summary = "Sketch → untextured 3D reference mesh to draw over"
    version = "0.1.0"

    InputModel = MeshInput
    OutputModel = MediaResponse  # binary .glb

    def model_loaders(self) -> dict[str, object]:
        return {TRELLIS_MODEL_KEY: load_trellis}

    def warmup(self, ctx: Context) -> None:
        """Run one tiny reconstruction so CUDA/spconv kernels initialise here.

        Measured on the live endpoint: a worker with warm *models* still took
        ~57s on its first real job and 4.1s on the next. That 14x cliff is
        kernel initialisation, and an artist should never be the one to pay it.

        Deduplicated by model identity in `warm_kernels`, because `reangle` may
        be sharing this worker and the same TRELLIS pipeline.
        """
        backbone = get_backbone("trellis", ctx.models.get(TRELLIS_MODEL_KEY))
        warm_kernels(backbone, TRELLIS_MODEL_KEY)

    def run(self, req: MeshInput, ctx: Context) -> MediaResponse:
        # Declared in model_loaders, so this is a dict lookup rather than the
        # ~14s pipeline load.
        backbone = get_backbone("trellis", ctx.models.get(TRELLIS_MODEL_KEY))
        result = run_pipeline(
            req.image,
            backbone=backbone,
            target_faces=req.target_faces,
            seed=req.seed,
        )
        return MediaResponse(
            data=result.glb,
            media_type="model/gltf-binary",
            filename="reference.glb",
        )
