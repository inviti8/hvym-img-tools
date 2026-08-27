"""`mesh` tool tests — no GPU, no TRELLIS.

The backbone is faked, so these exercise the parts we actually wrote: the
contract, decimation, and the cache-key behaviour a library depends on.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import trimesh
from PIL import Image

from hvym_img_tools.tools.mesh.pipeline import TARGET_FACES_DEFAULT, run_pipeline
from hvym_img_tools.tools.mesh.tool import MeshInput, MeshTool


def _png(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 200, 200)).save(buf, format="PNG")
    return buf.getvalue()


class FakeBackbone:
    """Returns a dense sphere, so decimation has real work to do."""

    def __init__(self, subdivisions: int = 4):
        self.seen_seed = None
        self._mesh = trimesh.creation.icosphere(subdivisions=subdivisions)

    def reconstruct(self, image, *, seed: int = 0, **_):
        self.seen_seed = seed
        return self._mesh.copy()


# --- contract --------------------------------------------------------------

def test_tool_is_registered_alongside_reangle():
    from hvym_img_tools.core import registry

    registry.discover()
    assert "mesh" in registry.names()
    assert "reangle" in registry.names(), "must not displace the existing tool"


def test_defaults_match_the_measured_recommendation():
    """20k is where every test subject landed at ~0.3 MB and >=0.994 silhouette."""
    assert MeshInput.model_fields["target_faces"].default == TARGET_FACES_DEFAULT == 20_000
    assert MeshInput(image=b"x").seed == 0, "a random seed would never hit the cache"


def test_target_faces_is_bounded():
    with pytest.raises(Exception):
        MeshInput(image=b"x", target_faces=500)        # below the floor
    with pytest.raises(Exception):
        MeshInput(image=b"x", target_faces=500_000)    # above the ceiling


def test_mesh_needs_trellis_not_triposr():
    assert MeshTool().models_needed() == ["trellis"]


# --- pipeline --------------------------------------------------------------

def test_decimates_to_the_target_and_returns_a_glb():
    fake = FakeBackbone()
    res = run_pipeline(_png(), backbone=fake, target_faces=2_000)
    assert res.faces_raw > res.faces_out, "dense input should have been decimated"
    assert res.faces_out <= 2_000
    assert res.glb[:4] == b"glTF"
    assert {"reconstruct", "decimate", "export"} <= set(res.timings)


def test_small_mesh_is_left_alone():
    """Decimation must not inflate or mangle geometry already under target."""
    fake = FakeBackbone(subdivisions=1)
    res = run_pipeline(_png(), backbone=fake, target_faces=200_000)
    assert res.faces_out == res.faces_raw


def test_result_is_untextured():
    """The whole licensing and cost argument rests on shipping no texture."""
    res = run_pipeline(_png(), backbone=FakeBackbone(), target_faces=2_000)
    loaded = trimesh.load(io.BytesIO(res.glb), file_type="glb", process=False)
    m = loaded if isinstance(loaded, trimesh.Trimesh) else list(loaded.geometry.values())[0]
    assert getattr(m.visual, "uv", None) is None or len(getattr(m.visual, "uv", []) or []) == 0


def test_seed_reaches_the_backbone():
    """A nondeterministic backbone would make a content-addressed cache lie."""
    fake = FakeBackbone()
    run_pipeline(_png(), backbone=fake, target_faces=2_000, seed=1234)
    assert fake.seen_seed == 1234


# --- caching, which the library depends on ---------------------------------

def test_params_change_the_cache_key():
    tool = MeshTool()
    base = tool.cache_key_parts(MeshInput(image=b"same"))
    for changed in (
        MeshInput(image=b"same", target_faces=50_000),
        MeshInput(image=b"same", seed=7),
        MeshInput(image=b"different"),
    ):
        assert tool.cache_key_parts(changed) != base


def test_same_request_is_a_stable_key():
    """The library keys on this: one sketch must always mean one asset."""
    tool = MeshTool()
    a = tool.cache_key_parts(MeshInput(image=b"same", target_faces=20_000, seed=0))
    b = tool.cache_key_parts(MeshInput(image=b"same", target_faces=20_000, seed=0))
    assert a == b
