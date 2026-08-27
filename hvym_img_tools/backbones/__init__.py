"""Image → 3D backbones, shared across tools.

Lifted out of `tools/reangle` once a second tool needed the same models: a tool
must not import another tool (AGENTS.md), and TRELLIS is expected to serve both
`mesh` and, eventually, `reangle` — `ReangleInput` already carries a `backbone`
field for exactly that substitution.

**CPU-importable by design.** Every torch/ML import lives inside a function, so
the registry and the tests still work on a box with no GPU stack.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol

from PIL import Image


class Backbone(Protocol):
    """What a reconstruction backbone must provide."""

    def reconstruct(self, image: Image.Image, **kwargs: Any) -> Any:
        """RGB image → a `trimesh.Trimesh`."""


_BACKBONES: dict[str, Callable[[Any], Backbone]] = {}


def register_backbone(name: str, factory: Callable[[Any], Backbone]) -> None:
    _BACKBONES[name] = factory


def get_backbone(name: str, model: Any) -> Backbone:
    try:
        return _BACKBONES[name](model)
    except KeyError:
        raise KeyError(f"unknown backbone {name!r}; available: {sorted(_BACKBONES)}") from None


def backbone_names() -> list[str]:
    return sorted(_BACKBONES)
