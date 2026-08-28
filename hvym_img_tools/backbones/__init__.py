"""Image → 3D backbones, shared across tools.

Lifted out of `tools/reangle` once a second tool needed the same models: a tool
must not import another tool (AGENTS.md), and TRELLIS is expected to serve both
`mesh` and, eventually, `reangle` — `ReangleInput` already carries a `backbone`
field for exactly that substitution.

**CPU-importable by design.** Every torch/ML import lives inside a function, so
the registry and the tests still work on a box with no GPU stack.

## The input convention belongs to the backbone

Callers hand over the **matted RGBA image** when they have one and plain RGB
otherwise; each backbone adapts. This is not cosmetic — the two backbones want
opposite things:

- TripoSR expects RGB flattened onto 0.5 grey (`imageio.composite_on`).
- TRELLIS branches on alpha: RGBA with any transparency is used as-is, while
  anything opaque gets re-segmented with its own rembg. Handing it a
  grey-composited RGB therefore *discards the isnet matte* and rebuilds a
  different silhouette — and the reangle UV bake aligns the artist's art to the
  isnet one, so the two must not diverge.

Doing this inside each backbone keeps the pipelines from having to know which
one they hold.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol

from PIL import Image


class Backbone(Protocol):
    """What a reconstruction backbone must provide."""

    def reconstruct(self, image: Image.Image, **kwargs: Any) -> Any:
        """RGBA (preferred) or RGB image → a `trimesh.Trimesh`."""


_BACKBONES: dict[str, Callable[[Any], Backbone]] = {}
_MODEL_KEYS: dict[str, str] = {}
_TARGET_FACES: dict[str, int | None] = {}


def register_backbone(
    name: str,
    factory: Callable[[Any], Backbone],
    *,
    model_key: str | None = None,
    default_target_faces: int | None = None,
) -> None:
    """Register a backbone, the `ModelCache` key it expects, and how much
    geometry it returns.

    `model_key` defaults to the backbone name, which is true for both shipped
    backbones. It exists so a caller can resolve *which* warm model to hand a
    named backbone — without it, `get_backbone("trellis", ...)` was being given
    whatever model the caller happened to hard-code.

    `default_target_faces` is how far a caller should decimate when the request
    does not say. It is a property of the backbone, not of the tool: TripoSR
    returns a ~30k-face depth proxy that wants no cap, TRELLIS returns
    176k-1.2M faces and must be capped. `None` means "leave it alone".
    """
    _BACKBONES[name] = factory
    _MODEL_KEYS[name] = model_key or name
    _TARGET_FACES[name] = default_target_faces


def get_backbone(name: str, model: Any) -> Backbone:
    try:
        return _BACKBONES[name](model)
    except KeyError:
        raise KeyError(f"unknown backbone {name!r}; available: {sorted(_BACKBONES)}") from None


def model_key_for(name: str) -> str:
    """The `ModelCache` key a named backbone needs.

    Pair this with `get_backbone` rather than passing a fixed key: a backbone
    handed the wrong model fails deep inside someone else's library, with an
    error that names neither the backbone nor the mistake.
    """
    try:
        return _MODEL_KEYS[name]
    except KeyError:
        raise KeyError(f"unknown backbone {name!r}; available: {sorted(_BACKBONES)}") from None


def default_target_faces_for(name: str) -> int | None:
    """How far to decimate this backbone's output when the request is silent."""
    return _TARGET_FACES.get(name)


def backbone_names() -> list[str]:
    return sorted(_BACKBONES)


#: `ModelCache` keys whose CUDA kernels have already been initialised.
#: Loading weights is not being ready -- spconv and CUDA compile or select
#: algorithms on the first real forward pass, which cost the mesh tool ~57s on
#: its first genuine job (docs/tools/mesh.md §6b). Two tools now share one
#: TRELLIS pipeline, so this keeps the startup pass to one per model rather than
#: one per tool.
#:
#: Keyed by the cache key, not by `id(model)`: CPython reuses the id of a
#: collected object, and a stale id silently skipped a warm-up that was needed.
_KERNELS_WARMED: set[str] = set()


def warm_kernels(backbone: Backbone, model_key: str, *, seed: int = 0) -> bool:
    """Run one tiny reconstruction, at most once per model key.

    Returns True if it actually ran. A solid shape rather than a blank canvas --
    the backbones matte their input, and an empty image gives them no foreground
    to find.
    """
    if model_key in _KERNELS_WARMED:
        return False

    from PIL import ImageDraw

    img = Image.new("RGB", (256, 256), (255, 255, 255))
    ImageDraw.Draw(img).ellipse((64, 64, 192, 192), fill=(90, 90, 90))
    backbone.reconstruct(img, seed=seed)
    _KERNELS_WARMED.add(model_key)
    return True
