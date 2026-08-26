"""Core framework tests. CPU-only, no GPU, no network (AGENTS.md §6)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from pydantic import BaseModel

from hvym_img_tools.core import registry
from hvym_img_tools.core.cache import ResultCache, hash_parts
from hvym_img_tools.core.config import Config
from hvym_img_tools.core.imageio import fit_to_frame
from hvym_img_tools.core.models import ModelCache
from hvym_img_tools.core.tool import Context, FileBytes, MediaResponse, Tool


class EchoInput(BaseModel):
    image: FileBytes
    scale: int = 2


class EchoTool(Tool):
    name = "_echo"
    summary = "test double"
    InputModel = EchoInput
    OutputModel = MediaResponse

    def run(self, req, ctx):
        return MediaResponse(data=req.image * req.scale, media_type="application/octet-stream")


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.unregister(EchoTool.name)
    yield
    registry.unregister(EchoTool.name)


# --- hashing ---------------------------------------------------------------

def test_hash_is_length_prefixed():
    """Without length prefixing these collide, silently serving the wrong cache entry."""
    assert hash_parts([b"ab", b"c"]) != hash_parts([b"a", b"bc"])


def test_hash_is_stable():
    assert hash_parts([b"x", b"y"]) == hash_parts([b"x", b"y"])


# --- cache -----------------------------------------------------------------

def test_cache_roundtrip(tmp_path):
    cache = ResultCache(tmp_path)
    assert cache.get("deadbeef") is None
    cache.put("deadbeef", b"payload", "model/gltf-binary")
    entry = cache.get("deadbeef")
    assert entry is not None
    assert entry.read() == b"payload"
    assert entry.media_type == "model/gltf-binary"
    assert cache.stats()["entries"] == 1


def test_cache_overwrite_is_atomic(tmp_path):
    cache = ResultCache(tmp_path)
    cache.put("k" * 8, b"first")
    cache.put("k" * 8, b"second")
    assert cache.get("k" * 8).read() == b"second"
    assert not list(tmp_path.rglob("*.tmp")), "temp files must not survive"


# --- model cache -----------------------------------------------------------

def test_model_cache_loads_once():
    calls = []
    cache = ModelCache(device="cpu")
    cache.register("m", lambda dev: calls.append(dev) or object())
    a, b = cache.get("m"), cache.get("m")
    assert a is b
    assert calls == ["cpu"], "loader must run exactly once"


def test_model_cache_unknown_key_lists_options():
    cache = ModelCache()
    cache.register("known", lambda d: None)
    with pytest.raises(KeyError, match="known"):
        cache.get("missing")


def test_model_cache_rejects_duplicate_registration():
    cache = ModelCache()
    cache.register("m", lambda d: 1)
    with pytest.raises(ValueError, match="already registered"):
        cache.register("m", lambda d: 2)
    cache.register("m", lambda d: 2, replace=True)  # explicit replace is allowed


# --- registry --------------------------------------------------------------

def test_register_and_lookup():
    registry.register(EchoTool)
    assert "_echo" in registry.names()
    assert registry.get("_echo") is EchoTool


def test_register_rejects_duplicate_name():
    registry.register(EchoTool)

    class Clashing(EchoTool):
        pass

    with pytest.raises(ValueError, match="already registered"):
        registry.register(Clashing)


def test_register_rejects_non_tool():
    with pytest.raises(TypeError):
        registry.register(dict)  # type: ignore[arg-type]


def test_unknown_tool_error_lists_available():
    registry.register(EchoTool)
    with pytest.raises(KeyError, match="_echo"):
        registry.get("nope")


# --- tool contract ---------------------------------------------------------

def test_file_fields_detected():
    assert EchoTool.file_fields() == ("image",)
    assert EchoTool.returns_media() is True


def test_filebytes_validates_and_is_binary_in_schema():
    req = EchoInput(image=b"abc")
    assert isinstance(req.image, FileBytes)
    schema = EchoInput.model_json_schema()
    assert schema["properties"]["image"]["format"] == "binary"


def test_cache_key_changes_with_input_and_params():
    tool = EchoTool()
    base = hash_parts(tool.cache_key_parts(EchoInput(image=b"a", scale=2)))
    assert base != hash_parts(tool.cache_key_parts(EchoInput(image=b"b", scale=2)))
    assert base != hash_parts(tool.cache_key_parts(EchoInput(image=b"a", scale=3)))


def test_media_response_rejects_non_bytes():
    with pytest.raises(TypeError):
        MediaResponse(data="not bytes", media_type="text/plain")  # type: ignore[arg-type]


def test_tool_runs_through_context(tmp_path):
    config = Config.from_env()
    ctx = Context(
        models=ModelCache(),
        cache=ResultCache(tmp_path),
        workspace=tmp_path,
        config=config,
    )
    out = EchoTool().run(EchoInput(image=b"ab", scale=3), ctx)
    assert out.data == b"ababab"


# --- imageio ---------------------------------------------------------------

def test_fit_preserves_aspect():
    """A tall shape must stay tall. Independent per-axis normalisation stretches
    the projection and destroys silhouette alignment (BENCHMARK.md §5.6)."""
    pts = np.array([[0.0, 0.0], [1.0, 4.0]])
    out = fit_to_frame(pts, size=512, margin=0.0)
    width = abs(out[1, 0] - out[0, 0])
    height = abs(out[1, 1] - out[0, 1])
    assert height > width
    assert height / width == pytest.approx(4.0, rel=1e-6)


def test_fit_is_centred_and_in_frame():
    pts = np.random.default_rng(0).normal(size=(200, 2))
    out = fit_to_frame(pts, size=512, margin=0.08)
    assert out.min() >= 0 and out.max() <= 511
    assert out.mean(0) == pytest.approx([255.5, 255.5], abs=40)


def test_config_from_env_is_cpu_safe(monkeypatch):
    monkeypatch.setenv("HVYM_DEVICE", "cpu")
    assert Config.from_env().resolve_device() == "cpu"
