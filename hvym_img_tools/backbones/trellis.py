"""TRELLIS backbone — microsoft/TRELLIS, MIT code and MIT weights.

Chosen on measurement, not reputation (docs/benchmark/paint3d/FINDINGS.md
Follow-up 6). Against TripoSR on largest-connected-component:

    chair      TripoSR 57.7% (14 parts)  ->  TRELLIS  98.6% (4 parts)
    rock       TripoSR 93.4% (7 parts)   ->  TRELLIS 100.0% (2 parts)

TripoSR cannot do the job for thin structures: its implicit field does not join
a chair's members, and raising `mc_resolution` makes it *worse* because finer
extraction resolves more disconnected pieces out of the same field.

Costs, measured: 3.7-7.0 s per object plus a one-off ~14.2 s model load, and a
**16 GB VRAM floor**. Output is 176k-1.2M faces, which callers must decimate.

Every heavy import is inside a function so this module stays CPU-importable.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from PIL import Image

from . import register_backbone

log = logging.getLogger(__name__)

#: ModelCache key for the TRELLIS pipeline.
TRELLIS_MODEL_KEY = "trellis"

#: The published image-to-3D checkpoint. MIT-licensed weights.
TRELLIS_REPO_ID = os.environ.get("TRELLIS_REPO_ID", "microsoft/TRELLIS-image-large")


def _apply_compat_shims() -> None:
    """Reconcile TRELLIS with the versions we actually ship.

    TRELLIS pins an older xformers API. Rather than hold xformers back (and with
    it torch), alias the moved symbols — verified equivalent during evaluation.
    """
    try:
        import xformers.ops as xops
    except ImportError:  # pragma: no cover - xformers is a container dependency
        return
    bias = getattr(xops.fmha, "attn_bias", None)
    if bias is None:
        return
    for name in ("BlockDiagonalMask", "BlockDiagonalCausalMask", "LowerTriangularMask"):
        if not hasattr(xops.fmha, name) and hasattr(bias, name):
            setattr(xops.fmha, name, getattr(bias, name))


def load_trellis(device: str) -> Any:
    """ModelCache loader for TRELLIS. Imports torch lazily, on purpose.

    Requires the TRELLIS repo on `sys.path` (env `TRELLIS_PATH`, default
    `/workspace/TRELLIS`), matching the container layout.
    """
    import sys

    # Must be set before trellis is imported: the backends are read from env at
    # module import, not at call time.
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")

    repo = os.environ.get("TRELLIS_PATH", "/workspace/TRELLIS")
    if repo not in sys.path:
        sys.path.insert(0, repo)

    _apply_compat_shims()

    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
    except ImportError as exc:  # pragma: no cover - environment-dependent
        missing = getattr(exc, "name", None)
        detail = (
            f"missing dependency {missing!r} (required by TRELLIS, not by us)"
            if missing and missing.split(".")[0] != "trellis"
            else f"repo not found or not importable at {repo!r}"
        )
        raise RuntimeError(
            f"cannot import TRELLIS: {detail}. Underlying error: {exc}. "
            f"TRELLIS_PATH={repo!r}. Its setup.sh fails silently on several "
            "submodules; the image must install xformers, spconv-cu124, utils3d "
            "and kaolin explicitly, with numpy<2 (docs/tools/mesh.md §4)."
        ) from exc

    from huggingface_hub import snapshot_download

    # TRELLIS resolves its sub-checkpoints as relative "ckpts/..." paths and only
    # treats them as local when they exist relative to the CWD. Downloading the
    # snapshot and loading from inside it avoids huggingface_hub mistaking those
    # for repo ids and 401ing.
    local = snapshot_download(TRELLIS_REPO_ID)
    cwd = os.getcwd()
    try:
        os.chdir(local)
        pipe = TrellisImageTo3DPipeline.from_pretrained(local)
    finally:
        os.chdir(cwd)

    if device.startswith("cuda"):
        pipe.cuda()
    log.info("TRELLIS loaded on %s from %s", device, TRELLIS_REPO_ID)
    return pipe


class TrellisBackbone:
    """Wraps a warm TRELLIS pipeline as a `Backbone`."""

    def __init__(self, model: Any) -> None:
        self._pipe = model

    def reconstruct(self, image: Image.Image, *, seed: int = 0, **_: Any) -> Any:
        """Sketch → `trimesh.Trimesh`, untextured.

        `seed` is threaded through so a request is reproducible: the result cache
        is content-addressed, and a nondeterministic backbone would make the same
        sketch return a different mesh on a cache miss.
        """
        import numpy as np
        import torch
        import trimesh

        out = self._pipe.run(image.convert("RGB"), seed=seed, formats=["mesh"])
        meshes = out.get("mesh") or []
        if not meshes:
            raise RuntimeError("TRELLIS returned no mesh")
        m = meshes[0]

        def _np(x: Any) -> Any:
            return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

        return trimesh.Trimesh(vertices=_np(m.vertices), faces=_np(m.faces), process=False)


register_backbone("trellis", TrellisBackbone)
