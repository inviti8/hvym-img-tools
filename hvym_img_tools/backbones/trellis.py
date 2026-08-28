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

Input convention: hand it the **matted RGBA** image when you have one. TRELLIS
uses a supplied alpha channel directly and only runs its own rembg when the
image is opaque, so preserving our matte both skips a redundant segmentation
and keeps the silhouette identical to the one reangle's UV bake aligns to.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from PIL import Image

from ..core.meshops import TARGET_FACES_DEFAULT
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


def _as_trellis_input(image: Image.Image) -> Image.Image:
    """RGBA carrying real transparency passes through; anything else → RGB.

    `TrellisImageTo3DPipeline.preprocess_image` treats an RGBA input as
    already-matted and skips rembg, but only when the alpha is not uniformly
    opaque. A fully-opaque RGBA would take the rembg branch regardless, so it is
    converted down rather than left implying an alpha that carries no
    information.
    """
    if image.mode == "RGBA":
        low, _high = image.getchannel("A").getextrema()
        if low < 255:
            return image
    return image.convert("RGB")


class TrellisBackbone:
    """Wraps a warm TRELLIS pipeline as a `Backbone`."""

    def __init__(self, model: Any) -> None:
        self._pipe = model

    def reconstruct(self, image: Image.Image, *, seed: int = 0, **_: Any) -> Any:
        """Sketch → `trimesh.Trimesh`, untextured.

        `seed` is threaded through so a request is reproducible: the result cache
        is content-addressed, and a nondeterministic backbone would make the same
        sketch return a different mesh on a cache miss.

        Surplus keyword arguments are ignored on purpose. `mc_resolution` is one
        of them: it is TripoSR's marching-cubes grid and means nothing here,
        because TRELLIS decodes a structured latent rather than extracting an
        isosurface. Face count is controlled by decimating afterwards
        (`core.meshops.decimate`), not by a knob on the model.
        """
        import numpy as np
        import torch
        import trimesh

        mesh, _ = self._run(image, seed=seed, formats=["mesh"])
        return mesh

    def reconstruct_appearance(self, image: Image.Image, *, seed: int = 0, **_: Any):
        """Sketch -> `(trimesh.Trimesh, GaussianCloud)`.

        The same reconstruction as `reconstruct`, plus the appearance TRELLIS
        already decoded and we normally discard. Asking for the Gaussian adds
        only the SLAT decoder pass -- it needs spconv, which we ship, and *not*
        a rasteriser, which we deliberately do not (`core.gsbake` explains why).

        Returned as plain arrays rather than TRELLIS's Gaussian object so
        nothing downstream has to import the model stack.
        """
        return self._run(image, seed=seed, formats=["mesh", "gaussian"])

    def _run(self, image: Image.Image, *, seed: int, formats: list[str]):
        import numpy as np
        import torch
        import trimesh

        from ..core.gsbake import GaussianCloud

        out = self._pipe.run(_as_trellis_input(image), seed=seed, formats=formats)
        meshes = out.get("mesh") or []
        if not meshes:
            raise RuntimeError("TRELLIS returned no mesh")
        m = meshes[0]

        def _np(x: Any) -> Any:
            return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

        mesh = trimesh.Trimesh(vertices=_np(m.vertices), faces=_np(m.faces), process=False)

        if "gaussian" not in formats:
            return mesh, None

        clouds = out.get("gaussian") or []
        if not clouds:
            raise RuntimeError("TRELLIS returned no gaussian despite being asked for one")
        g = clouds[0]

        # Use the public getters: they apply the aabb denormalisation and the
        # opacity sigmoid, which the raw `_`-prefixed tensors have not had.
        xyz = _np(g.get_xyz).reshape(-1, 3)
        opacity = _np(g.get_opacity).reshape(-1)
        try:
            dc = _np(g.get_features)
        except Exception:  # pragma: no cover - only when _features_rest is absent
            dc = _np(g._features_dc)
        dc = dc.reshape(len(xyz), -1)[:, :3]

        cloud = GaussianCloud(xyz=xyz, features_dc=dc, opacity=opacity)
        log.info(
            "TRELLIS appearance: %d gaussians, extent %s; mesh extent %s",
            len(cloud), cloud.extent().round(3).tolist(),
            (mesh.vertices.max(0) - mesh.vertices.min(0)).round(3).tolist(),
        )
        return mesh, cloud


register_backbone(
    "trellis",
    TrellisBackbone,
    model_key=TRELLIS_MODEL_KEY,
    # 176k-1.2M faces raw, up to 20.8 MB of .glb. At 20k every measured
    # subject kept 0.994-0.997 of its silhouette at ~0.3 MB, and on the
    # reangle path decimating 176,724 -> 20,000 cost 0.001 silhouette IoU.
    default_target_faces=TARGET_FACES_DEFAULT,
)
