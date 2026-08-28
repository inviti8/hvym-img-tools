"""Front-projected UV bake → embedded-texture `.glb`.

The mesh must carry a UV atlas of the artist's **original** drawing, never the
backbone's own predicted texture — that is the soft, style-lost version and must
never appear on the style path (REANGLE_PIPELINE.md §7.4).

The front axis is detected empirically, not assumed. TripoSR's happens to be +X
(image plane = Y,Z), and assuming the conventional (X,Y) silently projects the
art onto the mesh **from the side** — a bug this module exists to prevent
(BENCHMARK.md §5.5).
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ...core.imageio import DEFAULT_MARGIN, DEFAULT_SIZE, fit_to_frame

log = logging.getLogger(__name__)


@dataclass(slots=True)
class FrontView:
    """Which mesh axes face the camera, and how well the projection lines up."""

    h_axis: int          # mesh axis mapped to image x
    v_axis: int          # mesh axis mapped to image y
    d_axis: int          # mesh axis used as depth
    sign: int            # +1/-1 flip on the horizontal axis
    silhouette_iou: float

    @property
    def plane(self) -> tuple[int, int]:
        return self.h_axis, self.v_axis


def _splat_mask(points: np.ndarray, size: int, margin: float, radius: int = 2) -> np.ndarray:
    """Cheap dilated point-splat silhouette — for axis scoring only.

    Deliberately not a full raster: this runs 6× and only needs to rank
    candidates, so a dense z-buffer here would be wasted work.
    """
    px = fit_to_frame(points, size=size, margin=margin).astype(np.int32)
    mask = np.zeros((size, size), bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            mask[np.clip(px[:, 1] + dy, 0, size - 1), np.clip(px[:, 0] + dx, 0, size - 1)] = True
    return mask


#: Below this many points the vertex splat is too sparse to approximate a
#: silhouette, so we densify by sampling the faces first.
MIN_SPLAT_POINTS = 40_000


def densify(points: np.ndarray, faces: np.ndarray | None, target: int) -> np.ndarray:
    """Scatter extra points across faces so a splat approximates a silhouette.

    A coarse mesh (a cube has 8 vertices) otherwise scores ~0 IoU on every axis
    and the "best" view becomes arbitrary. Cheap barycentric sampling, no deps.

    Dimension-generic: `points` may be the 3D vertices or any per-vertex 2D
    attribute sharing their indexing, such as UVs. Anything scoring a mesh
    against the matte must densify the same way or the two masks are not
    comparable -- a decimated 20k-face mesh has ~10k vertices, well under
    `MIN_SPLAT_POINTS`, and splatting those alone understates coverage badly.
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


def detect_front_view(
    vertices: np.ndarray,
    alpha: np.ndarray,
    faces: np.ndarray | None = None,
    *,
    size: int = DEFAULT_SIZE,
    margin: float = DEFAULT_MARGIN,
) -> FrontView:
    """Pick the axis-aligned view whose silhouette best matches the artist's matte.

    Backbone-agnostic: swap TripoSR for InstantMesh and this still finds the
    front instead of inheriting a hard-coded convention. Pass `faces` so coarse
    meshes are densified before scoring.
    """
    # The matte may be baked at any texture resolution (2048 by default), but
    # front detection is a coarse silhouette-overlap score that gains nothing
    # from it -- and splatting at 2048 would be 16x the work. Match the mask's
    # own size so the two are comparable, and so this keeps producing exactly
    # the scores BENCHMARK.md measured at 512.
    if alpha.shape[:2] != (size, size):
        alpha = np.asarray(
            Image.fromarray(alpha).resize((size, size), Image.BILINEAR)
        )
    silhouette = alpha > 12
    vertices = densify(vertices, faces, MIN_SPLAT_POINTS)
    best: FrontView | None = None
    for d_axis in range(3):
        others = [i for i in range(3) if i != d_axis]
        for sign in (1, -1):
            plane = vertices[:, others].copy()
            if sign < 0:
                plane[:, 0] = -plane[:, 0]
            mask = _splat_mask(plane, size, margin)
            union = (mask | silhouette).sum()
            iou = float((mask & silhouette).sum() / union) if union else 0.0
            if best is None or iou > best.silhouette_iou:
                best = FrontView(
                    h_axis=others[0], v_axis=others[1], d_axis=d_axis,
                    sign=sign, silhouette_iou=iou,
                )
    assert best is not None
    log.info(
        "front view: image=(axis%d, axis%d) depth=axis%d sign=%+d IoU=%.3f",
        best.h_axis, best.v_axis, best.d_axis, best.sign, best.silhouette_iou,
    )
    return best


def front_planar_uv(
    vertices: np.ndarray,
    view: FrontView,
    *,
    size: int = DEFAULT_SIZE,
    margin: float = DEFAULT_MARGIN,
) -> np.ndarray:
    """UVs that sample the original art as seen from the front.

    REANGLE_PIPELINE.md §7.5 "simplest": front-facing geometry gets crisp art;
    back faces mirror-smear, which the angle window hides.
    """
    plane = vertices[:, [view.h_axis, view.v_axis]].copy()
    if view.sign < 0:
        plane[:, 0] = -plane[:, 0]
    px = fit_to_frame(plane, size=size, margin=margin)
    uv = px / (size - 1)
    uv[:, 1] = 1.0 - uv[:, 1]  # glTF UV origin is bottom-left
    return np.clip(uv, 0.0, 1.0)


def bake_glb(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, art: Image.Image) -> bytes:
    """Export an embedded-texture `.glb` so `load_from_memory` gets everything."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=art,
            metallicFactor=0.0,
            roughnessFactor=1.0,   # unlit-ish: never add fake 3D shading (§3)
        ),
    )
    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return buf.getvalue()
