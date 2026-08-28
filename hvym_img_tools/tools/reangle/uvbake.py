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

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ...core.imageio import DEFAULT_MARGIN, DEFAULT_SIZE, fit_to_frame
# Re-exported: both live in core so `core.gsbake` can share the sampler
# without core importing a tool. Callers still import them from here.
from ...core.meshops import MIN_SPLAT_POINTS, densify, textured_glb  # noqa: F401

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
    """Export an embedded-texture `.glb` so `load_from_memory` gets everything.

    Thin wrapper: the export itself moved to `core.meshops` once `mesh` needed
    the same one, including the unlit-ish material (§3 — never add fake 3D
    shading over artwork that already carries the artist's own).
    """
    return textured_glb(vertices, faces, uv, art)
