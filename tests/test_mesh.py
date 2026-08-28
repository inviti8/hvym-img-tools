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


# --- kernel warm-up --------------------------------------------------------
# Loading weights is not the same as being ready. On the live endpoint a worker
# with warm models still took ~57s on its first real job and 4.1s on the next;
# that 14x cliff is kernel initialisation, and it belongs at worker startup.


@pytest.fixture
def registered_backbone():
    """Swap the 'trellis' factory, then put the real one back.

    register_backbone mutates a module-level dict, so without this a test would
    leave a fake wired in for everything that ran after it.
    """
    from hvym_img_tools import backbones

    saved = (dict(backbones._BACKBONES), dict(backbones._MODEL_KEYS),
             dict(backbones._TARGET_FACES))
    # warm_kernels dedupes by model key for the life of the process, so without
    # clearing it the second test in a run silently gets no warm-up at all.
    backbones._KERNELS_WARMED.clear()
    calls: list[int] = []

    def install(reconstruct):
        class _B:
            # The registry calls factory(model), so the signature must take one.
            def __init__(self, model):
                self.model = model

            def reconstruct(self, image, *, seed: int = 0, **kw):
                calls.append(seed)
                return reconstruct(image, seed)

        backbones.register_backbone("trellis", _B)
        return calls

    yield install
    # Restore every dict register_backbone writes. Missing _TARGET_FACES left a
    # stub's `None` behind and quietly uncapped TRELLIS for the rest of the run.
    for live, original in zip(
        (backbones._BACKBONES, backbones._MODEL_KEYS, backbones._TARGET_FACES), saved
    ):
        live.clear()
        live.update(original)
    backbones._KERNELS_WARMED.clear()


class _Models:
    def get(self, key):
        return object()


def _ctx():
    from hvym_img_tools.core.tool import Context

    return Context(models=_Models(), cache=None, workspace=None, config=None)


def test_warmup_runs_a_real_reconstruction(registered_backbone):
    calls = registered_backbone(lambda img, seed: trimesh.creation.icosphere(subdivisions=1))
    MeshTool().warmup(_ctx())
    assert calls, "warmup must run a forward pass, not merely load weights"


def test_warmup_feeds_something_the_matte_can_find(registered_backbone):
    """A blank canvas gives the pipeline no foreground; the shape is deliberate."""
    seen = {}

    def capture(img, seed):
        seen["extrema"] = img.convert("L").getextrema()
        return trimesh.creation.icosphere(subdivisions=1)

    registered_backbone(capture)
    MeshTool().warmup(_ctx())
    lo, hi = seen["extrema"]
    assert lo != hi, "warm-up image must not be uniform"


def test_kernel_warmup_runs_once_per_model_not_once_per_tool(registered_backbone):
    """reangle and mesh share one TRELLIS pipeline; the startup pass is ~4s of
    GPU work and there is no reason to pay it twice."""
    calls = registered_backbone(lambda img, seed: trimesh.creation.icosphere(subdivisions=1))
    MeshTool().warmup(_ctx())
    MeshTool().warmup(_ctx())
    assert len(calls) == 1, f"expected one warm-up, got {len(calls)}"


def test_warmup_is_optional_on_the_base_contract():
    """A tool with no kernels to initialise must not be forced to implement it.

    Both shipped tools now do (they share TRELLIS), so this checks the contract
    itself rather than borrowing whichever tool happened not to override it.
    """
    from hvym_img_tools.core.tool import MediaResponse, Tool

    class _Bare(Tool):
        name = "bare"
        summary = "does nothing"
        InputModel = MeshInput
        OutputModel = MediaResponse

        def run(self, req, ctx):  # pragma: no cover - never called
            raise NotImplementedError

    assert _Bare().warmup(None) is None


def test_serverless_swallows_a_failed_warmup():
    """A worker that cannot warm up must still serve, just slower."""
    import inspect

    import hvym_img_tools.serverless as sl

    src = inspect.getsource(sl.init)
    assert "tool.warmup(ctx)" in src
    warm_block = src[src.index("tool.warmup(ctx)"):]
    assert "except Exception" in warm_block, "a failed warm-up must not kill the worker"
