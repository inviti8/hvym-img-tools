"""Mesh operations shared by more than one tool.

Lifted out of `tools/mesh` once `reangle` needed the same decimation: a tool
must not import another tool (AGENTS.md §2, §7).

Nothing here imports torch -- `core` stays CPU-importable.
"""
from __future__ import annotations

import logging

import numpy as np

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


def textured_glb(vertices, faces, uv, image) -> bytes:
    """Export an embedded-texture `.glb` so `load_from_memory` gets everything.

    `roughnessFactor=1.0` and no metallic is deliberate and applies to every
    caller: the texture is 2D artwork that already contains whatever shading the
    artist drew, so the renderer must never add its own on top.
    """
    import io

    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
    )
    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return buf.getvalue()


#: Below this many points the vertex splat is too sparse to approximate a
#: silhouette, so we densify by sampling the faces first.
MIN_SPLAT_POINTS = 40_000


def densify(points: np.ndarray, faces: np.ndarray | None, target: int) -> np.ndarray:
    """Scatter extra points across faces so a splat approximates a silhouette.

    A coarse mesh (a cube has 8 vertices) otherwise scores ~0 IoU on every axis
    and the "best" view becomes arbitrary. Cheap barycentric sampling, no deps.

    Dimension-generic: `points` may be the 3D vertices or any per-vertex
    attribute sharing their indexing -- UVs, colours, or several of those
    stacked together so one call keeps them in correspondence. Anything scoring
    a mesh against the matte must densify the same way or the two masks are not
    comparable -- a decimated 20k-face mesh has ~10k vertices, well under
    `MIN_SPLAT_POINTS`, and splatting those alone understates coverage badly.

    Lives here rather than in `reangle.uvbake` because `core.gsbake` needs the
    same barycentric sampling and core must not import a tool (AGENTS.md 2, 7).
    """
    if faces is None or len(faces) == 0 or len(points) >= target:
        return points
    per_face = int(np.ceil(target / len(faces)))
    rng = np.random.default_rng(0)  # deterministic: detection must be reproducible
    tris = points[faces]
    u = rng.random((len(faces), per_face, 1))
    v = rng.random((len(faces), per_face, 1))
    over = (u + v) > 1
    u = np.where(over, 1 - u, u)
    v = np.where(over, 1 - v, v)
    a, b, c = tris[:, 0:1], tris[:, 1:2], tris[:, 2:3]
    pts = (a + u * (b - a) + v * (c - a)).reshape(-1, points.shape[1])
    return np.vstack([points, pts])
