"""sketch → untextured reference mesh.

Deliberately short: reconstruct, decimate, export. There is no matte and no UV
bake, because the artist draws over this rather than looking at its surface.
Dropping texture is what keeps the tool MIT end to end (docs/tools/mesh.md §1).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ...backbones import Backbone
from ...core.imageio import decode_image
from ...core.meshops import TARGET_FACES_DEFAULT, decimate  # noqa: F401 - re-exported

log = logging.getLogger(__name__)

# TARGET_FACES_DEFAULT and the decimation itself moved to `core.meshops` once
# reangle needed them too -- a tool must not import another tool (AGENTS.md
# §2, §7). Re-exported here so this module's callers and tests are unchanged.


@dataclass(slots=True)
class MeshResult:
    glb: bytes
    faces_raw: int
    faces_out: int
    timings: dict[str, float] = field(default_factory=dict)
    #: Fraction of atlas texels a surface sample reached, when texturing.
    coverage: float = 0.0


def _texture_from_gaussian(mesh, cloud, texture_size: int):
    """Unwrap, then carry the Gaussian's colour onto the atlas.

    Unwrap *after* decimation: xatlas charts the faces it is given, so charting
    a 176k-face mesh and then throwing 89% of it away would waste the work and
    leave a shredded atlas.
    """
    from ...core.gsbake import bake_texture
    from ...core.meshops import textured_glb

    unwrapped = mesh.unwrap()  # xatlas; splits vertices along seams
    uv = np.asarray(unwrapped.visual.uv, float)
    verts = np.asarray(unwrapped.vertices, float)
    faces = np.asarray(unwrapped.faces)

    rgb, coverage = bake_texture(verts, faces, uv, cloud.field(), size=texture_size)
    art = Image.fromarray(rgb, "RGB")
    return textured_glb(verts, faces, uv, art), coverage


def run_pipeline(
    image_bytes: bytes,
    *,
    backbone: Backbone,
    target_faces: int = TARGET_FACES_DEFAULT,
    seed: int = 0,
    texture: str = "none",
    texture_size: int = 1024,
) -> MeshResult:
    """One sketch in, one untextured `.glb` out.

    The backbone is injected rather than loaded here so the heavy model comes
    from the shared `ModelCache` and this stays unit-testable with a fake.
    """
    timings: dict[str, float] = {}

    def _timed(label: str):
        class _T:
            def __enter__(self):
                self.t0 = time.perf_counter()
                return self

            def __exit__(self, *exc):
                timings[label] = round(time.perf_counter() - self.t0, 3)

        return _T()

    image = decode_image(image_bytes, "RGB")
    cloud = None
    with _timed("reconstruct"):
        if texture == "gaussian":
            if not hasattr(backbone, "reconstruct_appearance"):
                raise RuntimeError(
                    f"texture='gaussian' needs a backbone that decodes appearance; "
                    f"{type(backbone).__name__} only reconstructs geometry."
                )
            mesh, cloud = backbone.reconstruct_appearance(image, seed=seed)
        else:
            mesh = backbone.reconstruct(image, seed=seed)

    faces_raw = len(mesh.faces)
    with _timed("decimate"):
        mesh = decimate(mesh, target_faces)

    coverage = 0.0
    with _timed("export"):
        if cloud is not None:
            glb, coverage = _texture_from_gaussian(mesh, cloud, texture_size)
        else:
            glb = mesh.export(file_type="glb")
            if not isinstance(glb, bytes):  # trimesh may hand back a buffer
                glb = bytes(glb)

    log.info(
        "mesh: %d -> %d faces (%.1f%%), texture=%s coverage=%.3f, %d KB, %s",
        faces_raw, len(mesh.faces),
        len(mesh.faces) / max(1, faces_raw) * 100,
        texture, coverage, len(glb) // 1024, timings,
    )
    return MeshResult(
        glb=glb, faces_raw=faces_raw, faces_out=len(mesh.faces),
        timings=timings, coverage=coverage,
    )
