"""RunPod Serverless handler tests — no GPU, no weights, no `runpod` SDK.

The handler must never raise: a worker that dies on a malformed job stops
serving every subsequent job in the queue. These verify it returns
`{"error": ...}` for every bad-input path instead.
"""
from __future__ import annotations

import base64

import pytest
from pydantic import BaseModel, Field

from hvym_img_tools import serverless
from hvym_img_tools.core import registry
from hvym_img_tools.core.cache import ResultCache
from hvym_img_tools.core.config import Config
from hvym_img_tools.core.models import ModelCache
from hvym_img_tools.core.tool import Context, FileBytes, MediaResponse, Tool

RUNS: list[bytes] = []


class SlInput(BaseModel):
    image: FileBytes
    repeat: int = Field(default=1, ge=1, le=4)


class SlTool(Tool):
    name = "_sl"
    summary = "serverless test tool"
    version = "3.2.1"
    InputModel = SlInput
    OutputModel = MediaResponse

    def run(self, req: SlInput, ctx: Context) -> MediaResponse:
        RUNS.append(req.image)
        return MediaResponse(
            data=b"glTF" + req.image * req.repeat,
            media_type="model/gltf-binary",
            filename="out.glb",
        )


@pytest.fixture(autouse=True)
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("HVYM_DEVICE", "cpu")
    RUNS.clear()
    registry.unregister(SlTool.name)
    registry.register(SlTool)

    config = Config.from_env()
    config.cache_dir = tmp_path / "cache"
    config.workspace_dir = tmp_path / "work"
    config.ensure_dirs()
    ctx = Context(
        models=ModelCache(device="cpu"),
        cache=ResultCache(config.cache_dir),
        workspace=config.workspace_dir,
        config=config,
    )
    # Inject a prepared context so init() does no discovery or model loading.
    monkeypatch.setattr(serverless, "_CTX", ctx)
    monkeypatch.setattr(serverless, "_TOOLS", {SlTool.name: SlTool()})
    yield
    registry.unregister(SlTool.name)


def _job(**inp):
    return {"input": {"tool": "_sl", **inp}}


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def test_happy_path_returns_base64_media():
    out = serverless.handler(_job(image=b64(b"AB")))
    assert "error" not in out
    assert base64.b64decode(out["data"]) == b"glTFAB"
    assert out["media_type"] == "model/gltf-binary"
    assert out["filename"] == "out.glb"
    assert out["cached"] is False
    assert out["tool_version"] == "3.2.1"


def test_params_are_passed_through():
    out = serverless.handler(_job(image=b64(b"AB"), repeat=3))
    assert base64.b64decode(out["data"]) == b"glTFABABAB"


def test_cache_hit_skips_the_tool():
    first = serverless.handler(_job(image=b64(b"ZZ")))
    second = serverless.handler(_job(image=b64(b"ZZ")))
    assert first["data"] == second["data"]
    assert first["cached"] is False and second["cached"] is True
    assert len(RUNS) == 1, "a cached job must not re-run the tool"
    assert first["cache_key"] == second["cache_key"]


def test_different_params_miss_cache():
    serverless.handler(_job(image=b64(b"ZZ")))
    out = serverless.handler(_job(image=b64(b"ZZ"), repeat=2))
    assert out["cached"] is False
    assert len(RUNS) == 2


# --- the handler must never raise ------------------------------------------

def test_unknown_tool_returns_error():
    out = serverless.handler({"input": {"tool": "nope", "image": b64(b"x")}})
    assert "unknown tool" in out["error"]


def test_missing_required_file_returns_error():
    out = serverless.handler({"input": {"tool": "_sl"}})
    assert "missing required field" in out["error"]


def test_invalid_base64_returns_error():
    out = serverless.handler(_job(image="not!valid!base64!"))
    assert "not valid base64" in out["error"]


def test_non_string_binary_field_returns_error():
    out = serverless.handler(_job(image=12345))
    assert "must be a base64 string" in out["error"]


def test_validation_failure_returns_error():
    out = serverless.handler(_job(image=b64(b"AB"), repeat=99))  # exceeds le=4
    assert "invalid input" in out["error"]


def test_tool_exception_is_caught(monkeypatch):
    def boom(self, req, ctx):
        raise RuntimeError("backbone exploded")

    monkeypatch.setattr(SlTool, "run", boom)
    out = serverless.handler(_job(image=b64(b"AB")))
    assert "backbone exploded" in out["error"]


@pytest.mark.parametrize("job", [None, {}, {"input": None}, {"input": "nope"}, {"input": []}])
def test_malformed_jobs_do_not_raise(job):
    out = serverless.handler(job)
    assert isinstance(out, dict) and "error" in out
