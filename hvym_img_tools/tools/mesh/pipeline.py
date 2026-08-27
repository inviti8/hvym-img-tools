"""sketch → untextured reference mesh.

Deliberately short: reconstruct, decimate, export. There is no matte and no UV
bake, because the artist draws over this rather than looking at its surface.
Dropping texture is what keeps the tool MIT end to end (docs/tools/mesh.md §1).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ...backbones import Backbone
from ...core.imageio import decode_image

log = logging.getLogger(__name__)

#: Raw TRELLIS output is 176k-1.2M faces (up to 20.8 MB of .glb), which is not a
#: shippable payload. Measured: at 20k every test subject lands at ~0.3 MB while
#: keeping 0.994-0.997 of its silhouette -- smaller than a reangle result, which
#: matters for a tool whose point is accumulating a library.
TARGET_FACES_DEFAULT = 20_000


@dataclass(slots=True)
class MeshResult:
    glb: bytes
    faces_raw: int
    faces_out: int
    timings: dict[str, float] = field(default_factory=dict)


def _decimate(mesh, target_faces: int):
    """Quadric decimation, tolerant of trimesh's signature change across majors."""
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except TypeError:  # trimesh < 4 took a positional count
        return mesh.simplify_quadric_decimation(target_faces)


def run_pipeline(
    image_bytes: bytes,
    *,
    backbone: Backbone,
    target_faces: int = TARGET_FACES_DEFAULT,
    seed: int = 0,
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

    with _timed("reconstruct"):
        mesh = backbone.reconstruct(decode_image(image_bytes, "RGB"), seed=seed)

    faces_raw = len(mesh.faces)
    with _timed("decimate"):
        mesh = _decimate(mesh, target_faces)

    with _timed("export"):
        glb = mesh.export(file_type="glb")
        if not isinstance(glb, bytes):  # trimesh may hand back a buffer
            glb = bytes(glb)

    log.info(
        "mesh: %d -> %d faces (%.1f%%), %d KB, %s",
        faces_raw, len(mesh.faces),
        len(mesh.faces) / max(1, faces_raw) * 100,
        len(glb) // 1024, timings,
    )
    return MeshResult(glb=glb, faces_raw=faces_raw, faces_out=len(mesh.faces), timings=timings)
