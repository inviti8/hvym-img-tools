"""Mesh operations shared by more than one tool.

Lifted out of `tools/mesh` once `reangle` needed the same decimation: a tool
must not import another tool (AGENTS.md §2, §7).

Nothing here imports torch -- `core` stays CPU-importable.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

#: Raw TRELLIS output is 176k-1.2M faces (up to 20.8 MB of .glb). Measured: at
#: 20k every test subject lands at ~0.3 MB while keeping 0.994-0.997 of its
#: silhouette (docs/tools/mesh.md §3). TripoSR's ~31k is already under this, so
#: applying it to the existing reangle path is a no-op.
TARGET_FACES_DEFAULT = 20_000


def decimate(mesh, target_faces: int | None):
    """Quadric-decimate to at most `target_faces`. A no-op if already under.

    `None` or a non-positive count means "leave it alone", which is what the
    TripoSR path wants: its mesh is a depth proxy that nothing needs to shrink.

    Tolerant of trimesh's signature change across majors -- 4.x takes
    `face_count=`, 3.x took a positional count.
    """
    if not target_faces or target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except TypeError:  # trimesh < 4 took a positional count
        return mesh.simplify_quadric_decimation(target_faces)
