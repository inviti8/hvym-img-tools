"""The reangle pipeline: matte → reconstruct → front-projected UV bake → glb.

Measured end-to-end at ~1.78 s warm on an RTX 4090 (docs/BENCHMARK.md §1).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ...core.imageio import (
    DEFAULT_TEXTURE_SIZE,
    composite_on,
    decode_image,
    isnet_matte,
)
from . import reconstruct as backbones
from .uvbake import bake_glb, detect_front_view, front_planar_uv

log = logging.getLogger(__name__)

#: ModelCache key for the isnet matting model.
ISNET_MODEL_KEY = "isnet"


@dataclass(slots=True)
class ReangleResult:
    glb: bytes
    matte_png: bytes
    vertices: int
    faces: int
    silhouette_iou: float
    front_axis: int
    timings: dict[str, float] = field(default_factory=dict)


def run_pipeline(
    image_bytes: bytes,
    *,
    isnet_session,
    backbone: backbones.Backbone,
    mc_resolution: int = backbones.MC_RESOLUTION_DEFAULT,
    texture_size: int = DEFAULT_TEXTURE_SIZE,
) -> ReangleResult:
    """One drawing in, one textured `.glb` out.

    Dependencies are injected rather than loaded here so every heavy model comes
    from the shared `ModelCache` and this stays unit-testable with fakes.
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

    with _timed("matte"):
        matte = isnet_matte(decode_image(image_bytes, "RGB"), isnet_session, size=texture_size)

    with _timed("reconstruct"):
        mesh = backbone.reconstruct(composite_on(matte.image), mc_resolution=mc_resolution)

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)

    with _timed("uvbake"):
        # Detect the front rather than assuming it — see uvbake.detect_front_view.
        view = detect_front_view(vertices, matte.alpha, faces)
        uv = front_planar_uv(vertices, view)
        glb = bake_glb(vertices, faces, uv, matte.image)

    timings["total"] = round(sum(timings.values()), 3)
    log.info(
        "reangle done in %.3fs (%s) verts=%d faces=%d IoU=%.3f",
        timings["total"], timings, len(vertices), len(faces), view.silhouette_iou,
    )
    return ReangleResult(
        glb=glb,
        matte_png=matte.to_png(),
        vertices=len(vertices),
        faces=len(faces),
        silhouette_iou=round(view.silhouette_iou, 4),
        front_axis=view.d_axis,
        timings=timings,
    )


def load_isnet(device: str):
    """ModelCache loader for the isnet matting model.

    onnxruntime-gpu must match the CUDA major version: 1.29 is a CUDA-13 build
    and silently falls back to CPU on a CUDA-12 box, which made the matte 6.44 s
    instead of 0.027 s — 242× (BENCHMARK.md §5.2).
    """
    import os

    import onnxruntime as ort

    path = os.environ.get("ISNET_PATH", "/workspace/models/isnet_dis.onnx")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.startswith("cuda")
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(path, providers=providers)
    active = session.get_providers()
    if device.startswith("cuda") and "CUDAExecutionProvider" not in active:
        log.warning(
            "isnet fell back to CPU (providers=%s) — expect ~6.4s per matte instead "
            "of ~0.03s. Check onnxruntime-gpu matches the CUDA major version.",
            active,
        )
    return session
