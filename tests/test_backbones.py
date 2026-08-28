"""Backbone registry, shared mesh ops, and the input convention each backbone owns.

These cover three defects found while scoping the TripoSR → TRELLIS swap:

1. `ReangleTool.run` resolved the model with a hard-coded TripoSR key, so
   `backbone=trellis` handed a `TSR` object to `TrellisBackbone`.
2. The pipeline composited the matte onto grey before every backbone. That is
   TripoSR's convention; it destroys the alpha TRELLIS reads, sending it to
   rembg for a silhouette the UV bake does not agree with.
3. Decimation lived inside `tools/mesh`, where `reangle` may not import it.

CPU-only: no torch, no TRELLIS, no weights.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import trimesh
from PIL import Image

from hvym_img_tools import backbones
from hvym_img_tools.backbones import get_backbone, model_key_for, register_backbone
from hvym_img_tools.core.meshops import TARGET_FACES_DEFAULT, decimate


@pytest.fixture
def clean_registry():
    """register_backbone mutates module-level dicts; put them back afterwards."""
    saved = (dict(backbones._BACKBONES), dict(backbones._MODEL_KEYS),
             dict(backbones._TARGET_FACES))
    yield
    for live, original in zip(
        (backbones._BACKBONES, backbones._MODEL_KEYS, backbones._TARGET_FACES), saved
    ):
        live.clear()
        live.update(original)


# --- the registry knows which model each backbone wants ---------------------

def test_shipped_backbones_resolve_their_own_model_key():
    from hvym_img_tools.core import registry

    registry.discover()  # imports both tools, which register both backbones
    assert model_key_for("triposr") == "triposr"
    assert model_key_for("trellis") == "trellis"


def test_model_key_defaults_to_the_backbone_name(clean_registry):
    register_backbone("stub", lambda model: model)
    assert model_key_for("stub") == "stub"


def test_model_key_can_differ_from_the_backbone_name(clean_registry):
    register_backbone("stub", lambda model: model, model_key="shared-weights")
    assert model_key_for("stub") == "shared-weights"


def test_unknown_backbone_names_itself_and_the_alternatives():
    with pytest.raises(KeyError) as exc:
        model_key_for("wonder3d")
    assert "wonder3d" in str(exc.value)
    assert "triposr" in str(exc.value), "the error must list what IS available"


# --- the defect: reangle asked for the wrong model --------------------------

class _RecordingModels:
    def __init__(self, **models):
        self.asked: list[str] = []
        self._models = models

    def get(self, key):
        self.asked.append(key)
        return self._models.get(key, object())


def test_reangle_run_fetches_the_model_the_requested_backbone_needs(clean_registry):
    """Regression: `backbone=trellis` used to be handed TripoSR's weights."""
    from hvym_img_tools.core.tool import Context
    from hvym_img_tools.tools.reangle.tool import ReangleInput, ReangleTool

    seen = {}

    class _Fake:
        def __init__(self, model):
            seen["model"] = model

        def reconstruct(self, image, **kw):
            seen["kwargs"] = kw
            return trimesh.creation.icosphere(subdivisions=3)

    register_backbone("fake3d", _Fake, model_key="fake-weights")
    sentinel = object()
    models = _RecordingModels(**{"fake-weights": sentinel})

    ctx = Context(models=models, cache=None, workspace=None, config=None)
    with pytest.raises(Exception):
        # isnet is a stub object, so the matte fails -- by which point the
        # backbone has already been resolved, which is all this asserts.
        ReangleTool().run(ReangleInput(image=b"x", backbone="fake3d"), ctx)

    assert "fake-weights" in models.asked, f"asked for {models.asked}"
    assert "triposr" not in models.asked, "must not fetch the default backbone's model"
    assert seen["model"] is sentinel


# --- each backbone owns its input convention --------------------------------

def _rgba(alpha: int) -> Image.Image:
    img = Image.new("RGBA", (32, 32), (10, 200, 30, 255))
    img.putalpha(Image.new("L", (32, 32), alpha))
    return img


def test_trellis_keeps_a_real_matte():
    """A supplied alpha is what makes TRELLIS skip its own rembg."""
    from hvym_img_tools.backbones.trellis import _as_trellis_input

    out = _as_trellis_input(_rgba(128))
    assert out.mode == "RGBA", "the matte must survive; rembg would replace it"


def test_trellis_drops_an_alpha_that_carries_nothing():
    """Fully opaque RGBA takes TRELLIS's rembg branch anyway -- don't imply otherwise."""
    from hvym_img_tools.backbones.trellis import _as_trellis_input

    assert _as_trellis_input(_rgba(255)).mode == "RGB"


def test_trellis_accepts_plain_rgb():
    """The `mesh` tool passes RGB and relies on TRELLIS matting it."""
    from hvym_img_tools.backbones.trellis import _as_trellis_input

    assert _as_trellis_input(Image.new("RGB", (32, 32), (1, 2, 3))).mode == "RGB"


def test_triposr_composites_the_matte_onto_grey_itself():
    """TripoSR's convention moved into the backbone when the pipeline stopped
    imposing it. If this regresses, TripoSR silently sees a transparent input."""
    from hvym_img_tools.tools.reangle.reconstruct import _as_triposr_input

    out = _as_triposr_input(_rgba(0))  # fully transparent green
    assert out.mode == "RGB", "TripoSR must never receive RGBA"
    assert all(120 <= c <= 135 for c in out.getpixel((0, 0))), out.getpixel((0, 0))


def test_triposr_leaves_rgb_alone():
    from hvym_img_tools.tools.reangle.reconstruct import _as_triposr_input

    img = Image.new("RGB", (8, 8), (1, 2, 3))
    assert _as_triposr_input(img) is img


def test_backbones_tolerate_each_others_keywords():
    """The pipeline passes both `mc_resolution` and `seed` without knowing which
    backbone it holds, so neither may raise on the other's knob."""
    import inspect

    from hvym_img_tools.backbones.trellis import TrellisBackbone
    from hvym_img_tools.tools.reangle.reconstruct import TripoSRBackbone

    for cls in (TrellisBackbone, TripoSRBackbone):
        params = inspect.signature(cls.reconstruct).parameters.values()
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params), cls


# --- shared decimation ------------------------------------------------------

def test_decimate_reduces_a_dense_mesh():
    dense = trimesh.creation.icosphere(subdivisions=5)
    assert len(dense.faces) > 2_000
    assert len(decimate(dense, 2_000).faces) <= 2_000


def test_decimate_leaves_a_mesh_already_under_target():
    small = trimesh.creation.icosphere(subdivisions=1)
    assert decimate(small, TARGET_FACES_DEFAULT) is small


def test_decimate_none_means_no_cap():
    """TripoSR's depth proxy is passed through untouched -- `None`, not a big number."""
    dense = trimesh.creation.icosphere(subdivisions=5)
    assert decimate(dense, None) is dense
    assert decimate(dense, 0) is dense


def test_mesh_tool_still_reaches_the_shared_helper():
    """`tools/mesh` re-exports these; a tool must not import another tool."""
    from hvym_img_tools.tools.mesh import pipeline as mesh_pipeline

    assert mesh_pipeline.TARGET_FACES_DEFAULT == TARGET_FACES_DEFAULT == 20_000
    assert mesh_pipeline.decimate is decimate
