"""Single-image → 3D backbone. **Swappable by design.**

TripoSR (MIT, ~0.11 s/image measured) is the productization target. The prototype
used DrawingSpinUp (Wonder3D + NeuS) but **Wonder3D weights are CC-BY-NC → demo
only**, so it is not shipped here (REANGLE_PIPELINE.md §8, BENCHMARK.md §4).

Every torch/TripoSR import is deliberately inside a function: this module must be
importable on a CPU box with no ML stack so the registry and tests still work.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from PIL import Image

from ...backbones import (  # noqa: F401 - re-exported for callers
    Backbone,
    backbone_names,
    default_target_faces_for,
    get_backbone,
    model_key_for,
    register_backbone,
)
from ...core.imageio import composite_on

log = logging.getLogger(__name__)

#: ModelCache key for the TripoSR weights.
TRIPOSR_MODEL_KEY = "triposr"

#: Measured sweet spots (BENCHMARK.md §1). Extraction scales ~res³ and is ~74%
#: of wall-clock, so this is the one real cost lever. The mesh is only a depth
#: proxy (REANGLE_PIPELINE.md §4.3), so lower is likely fine — unvalidated.
MC_RESOLUTION_DEFAULT = 256


# The registry moved to `hvym_img_tools.backbones` once `mesh` needed the same
# models -- a tool must not import another tool. Re-exported here so this
# module's callers are unchanged, and so TripoSR and TRELLIS share one registry
# for the swap `ReangleInput.backbone` already anticipates.


# --------------------------------------------------------------------------- #
# TripoSR
# --------------------------------------------------------------------------- #

def load_triposr(device: str) -> Any:
    """ModelCache loader for TripoSR. Imports torch lazily, on purpose.

    Requires the TripoSR repo on `sys.path` (env `TRIPOSR_PATH`, default
    `/workspace/TripoSR`), matching the container layout.
    """
    import sys

    repo = os.environ.get("TRIPOSR_PATH", "/workspace/TripoSR")
    if repo not in sys.path:
        sys.path.insert(0, repo)

    try:
        from tsr.system import TSR
    except ImportError as exc:  # pragma: no cover - environment-dependent
        # Report the *actual* failure. A generic "not importable" message sent us
        # hunting a path problem when the real cause was a missing transitive
        # dependency (imageio) several imports deep inside tsr.
        missing = getattr(exc, "name", None)
        detail = (
            f"missing dependency {missing!r} (required by TripoSR, not by us)"
            if missing and missing.split(".")[0] != "tsr"
            else f"repo not found or not importable at {repo!r}"
        )
        raise RuntimeError(
            f"cannot import TripoSR: {detail}. Underlying error: {exc}. "
            f"TRIPOSR_PATH={repo!r}. Note transformers must be <5 — v5 renamed "
            "ViT weights and the checkpoint will not load (BENCHMARK.md §5.1)."
        ) from exc

    chunk = int(os.environ.get("TRIPOSR_CHUNK_SIZE", "8192"))
    model = TSR.from_pretrained(
        os.environ.get("TRIPOSR_REPO_ID", "stabilityai/TripoSR"),
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(chunk)
    model.to(device)
    log.info("TripoSR loaded on %s (chunk_size=%d)", device, chunk)
    return model


def _as_triposr_input(image: Image.Image) -> Image.Image:
    """RGBA matte → RGB flattened onto 0.5 grey; RGB passes through.

    TripoSR's `run.py` convention. It lives here rather than in the pipeline so
    the caller can hand every backbone the same matted RGBA and let each adapt
    -- TRELLIS wants the opposite (see backbones/__init__.py).
    """
    return composite_on(image) if image.mode == "RGBA" else image


class TripoSRBackbone:
    """Wraps a warm TripoSR model as a `Backbone`."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def reconstruct(
        self,
        image: Image.Image,
        *,
        mc_resolution: int = MC_RESOLUTION_DEFAULT,
        **_: Any,
    ) -> Any:
        """RGBA (preferred) or RGB in → `trimesh.Trimesh`.

        TripoSR wants RGB flattened onto 0.5 grey (its `run.py` convention), so
        the composite happens *here*. Callers pass the matte and each backbone
        adapts, because TRELLIS wants the opposite — see backbones/__init__.py.

        Surplus keywords (`seed`, `target_faces`) are ignored: TripoSR is
        deterministic and decimation is the pipeline's job.
        """
        import torch

        image = _as_triposr_input(image)
        device = next(self._model.parameters()).device
        with torch.no_grad():
            scene_codes = self._model([image], device=device)
            meshes = self._model.extract_mesh(scene_codes, True, resolution=mc_resolution)
        if not meshes:
            raise RuntimeError("TripoSR returned no mesh")
        return meshes[0]


register_backbone(
    "triposr",
    TripoSRBackbone,
    model_key=TRIPOSR_MODEL_KEY,
    # ~30k faces: a depth proxy that wants no cap. Capping it would
    # change the output of a path that has shipped since 0.1.0.
    default_target_faces=None,
)
