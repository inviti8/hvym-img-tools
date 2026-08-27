"""Shared image helpers: decode/encode, silhouette normalisation, isnet matting.

Generic on purpose — anything reangle-specific belongs in the tool, not here
(AGENTS.md §9). The matte pipeline is a faithful port of
`../infinipaint/scripts/reangle/prep_input.py`, whose exact normalisation the
downstream geometry depends on.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

#: prep_input.py constants. **DEFAULT_MARGIN** is what the UV alignment depends
#: on — `isnet_matte` pads by it and `fit_to_frame` mirrors that padding, so
#: drifting it invalidates the 0.776 silhouette IoU in docs/BENCHMARK.md §2.
#:
#: DEFAULT_SIZE is only the frame `fit_to_frame` maps into, and the result is
#: divided by `size - 1` to give normalised UVs — so it cancels out. It is safe
#: to raise the *texture* resolution without touching alignment; verified to
#: 2.2e-16 across 512-4096.
DEFAULT_SIZE = 512
DEFAULT_MARGIN = 0.08

#: Texture resolution for a baked result. 512 visibly softened the artist's
#: linework once it was magnified in Inkternity; the drawings are ~2K to begin
#: with, so this is at or below source resolution rather than an upscale. The
#: cost is small because linework compresses well — measured 342-634 KB of PNG
#: at 2048, against a ~680 KB mesh that already dominates the .glb.
#:
#: Note the *silhouette edge* does not sharpen past isnet's own 1024 input; what
#: this recovers is the interior linework, which comes from the source image.
DEFAULT_TEXTURE_SIZE = 2048
ALPHA_LO, ALPHA_HI = 0.30, 0.65
ALPHA_FLOOR = 12


def decode_image(data: bytes, mode: str = "RGB") -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert(mode)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not decode image ({type(exc).__name__}: {exc})") from exc


def encode_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@dataclass(slots=True)
class Matte:
    """A matted character: 512² RGBA, alpha = silhouette."""

    image: Image.Image
    bbox: tuple[int, int, int, int]
    source_size: tuple[int, int]

    @property
    def alpha(self) -> np.ndarray:
        return np.asarray(self.image)[:, :, 3]

    def to_png(self) -> bytes:
        return encode_png(self.image)


def isnet_matte(
    image: Image.Image,
    session: Any,
    *,
    size: int = DEFAULT_SIZE,
    margin: float = DEFAULT_MARGIN,
) -> Matte:
    """Segment the character, trim to silhouette, pad square, resize.

    `session` is an onnxruntime InferenceSession for isnet_dis.onnx. Passed in
    rather than loaded here so it comes from the shared `ModelCache`.
    """
    w0, h0 = image.size
    inp = session.get_inputs()[0]
    ih = inp.shape[2] if isinstance(inp.shape[2], int) else 1024
    iw = inp.shape[3] if isinstance(inp.shape[3], int) else 1024

    # isnet normalisation: mean 0.5, std 1.0, CHW — identical to mv.remove_background
    arr = np.asarray(image.resize((iw, ih), Image.BILINEAR), dtype=np.float32)
    arr = np.transpose(arr / 255.0 - 0.5, (2, 0, 1))[None].astype(np.float32)
    mask = session.run(None, {inp.name: arr})[0][0][0]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    full = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((w0, h0), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    alpha = (np.clip((full - ALPHA_LO) / (ALPHA_HI - ALPHA_LO), 0.0, 1.0) * 255).astype(np.uint8)

    rgba = np.dstack([np.asarray(image, dtype=np.uint8), alpha])
    ys, xs = np.where(alpha > ALPHA_FLOOR)
    if xs.size == 0:
        raise ValueError("empty matte — isnet found no foreground in the image")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

    crop = Image.fromarray(rgba).crop((x0, y0, x1 + 1, y1 + 1))
    cw, ch = crop.size
    side = int(round(max(cw, ch) * (1 + 2 * margin)))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(crop, ((side - cw) // 2, (side - ch) // 2), crop)

    return Matte(
        image=canvas.resize((size, size), Image.LANCZOS),
        bbox=(x0, y0, x1, y1),
        source_size=(w0, h0),
    )


def composite_on(image: Image.Image, value: float = 0.5) -> Image.Image:
    """Flatten RGBA onto a constant grey — TripoSR's expected input convention."""
    arr = np.asarray(image).astype(np.float32) / 255.0
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + value * (1 - arr[:, :, 3:4])
    return Image.fromarray((rgb * 255).astype(np.uint8))


def fit_to_frame(
    points: np.ndarray,
    *,
    size: int = DEFAULT_SIZE,
    margin: float = DEFAULT_MARGIN,
    flip_y: bool = True,
) -> np.ndarray:
    """Aspect-preserving map of 2D points into a `size`² frame, centred.

    Mirrors `isnet_matte`'s padding so a mesh projection lands on the matte's
    silhouette. Aspect MUST be preserved — normalising each axis independently
    stretches the projection and destroys the alignment (see BENCHMARK.md §5.6).
    """
    mn, mx = points.min(0), points.max(0)
    long_side = float(max(mx - mn)) * (1 + 2 * margin)
    out = (points - (mn + mx) / 2.0) * ((size - 1) / (long_side + 1e-9)) + (size - 1) / 2.0
    if flip_y:
        out[:, 1] = (size - 1) - out[:, 1]
    return out
