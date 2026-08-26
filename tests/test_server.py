"""HTTP layer tests — the full request path with a stub tool, no GPU, no weights.

Proves the framework contract end to end: multipart in, media out, OpenAPI
generated, and the input-hash cache actually short-circuits a second call.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from hvym_img_tools.core import registry
from hvym_img_tools.core.config import Config
from hvym_img_tools.core.server import create_app
from hvym_img_tools.core.tool import Context, FileBytes, MediaResponse, Tool

CALLS: list[bytes] = []


class StubInput(BaseModel):
    image: FileBytes = Field(description="an image")
    repeat: int = Field(default=1, ge=1, le=8)


class StubTool(Tool):
    name = "_stub"
    summary = "stub tool for HTTP tests"
    version = "9.9.9"
    InputModel = StubInput
    OutputModel = MediaResponse

    def models_needed(self) -> list[str]:
        return []

    def run(self, req: StubInput, ctx: Context) -> MediaResponse:
        CALLS.append(req.image)
        return MediaResponse(
            data=b"glTF" + req.image * req.repeat,
            media_type="model/gltf-binary",
            filename="stub.glb",
        )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HVYM_DEVICE", "cpu")
    CALLS.clear()
    registry.unregister(StubTool.name)
    registry.register(StubTool)
    config = Config.from_env()
    config.cache_dir = tmp_path / "cache"
    config.workspace_dir = tmp_path / "work"
    # discover=False so a half-installed real tool can't affect these tests
    with TestClient(create_app(config, discover=False)) as c:
        yield c
    registry.unregister(StubTool.name)


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert "_stub" in body["tools"]
    assert body["device"] == "cpu"


def test_list_tools(client):
    body = client.get("/tools").json()
    entry = next(t for t in body["tools"] if t["name"] == "_stub")
    assert entry["version"] == "9.9.9"
    assert entry["file_fields"] == ["image"]
    assert entry["returns_media"] is True


def test_openapi_documents_multipart_upload(client):
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/tools/_stub"]["post"]
    content = op["requestBody"]["content"]
    assert "multipart/form-data" in content

    schema = content["multipart/form-data"]["schema"]
    if "$ref" in schema:  # FastAPI hoists the body model into components
        schema = spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
    props = schema["properties"]

    # OpenAPI 3.1 marks binary with contentMediaType, not `format: binary`
    assert props["image"].get("contentMediaType") or props["image"].get("format") == "binary"
    assert schema["required"] == ["image"]
    assert props["repeat"]["type"] == "integer"
    assert props["repeat"]["default"] == 1


def test_post_returns_media_with_headers(client):
    resp = client.post("/tools/_stub", files={"image": ("d.png", b"AB", "image/png")})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "model/gltf-binary"
    assert resp.headers["x-tool-version"] == "9.9.9"
    assert resp.headers["x-cache"] == "MISS"
    assert "stub.glb" in resp.headers["content-disposition"]
    assert resp.content == b"glTFAB"


def test_form_field_is_applied(client):
    resp = client.post(
        "/tools/_stub",
        files={"image": ("d.png", b"AB", "image/png")},
        data={"repeat": "3"},
    )
    assert resp.content == b"glTFABABAB"


def test_cache_short_circuits_second_call(client):
    files = {"image": ("d.png", b"XY", "image/png")}
    first = client.post("/tools/_stub", files=files)
    second = client.post("/tools/_stub", files={"image": ("d.png", b"XY", "image/png")})
    assert first.content == second.content
    assert first.headers["x-cache"] == "MISS"
    assert second.headers["x-cache"] == "HIT"
    assert len(CALLS) == 1, "cached request must not re-run the tool"


def test_different_params_miss_the_cache(client):
    client.post("/tools/_stub", files={"image": ("d.png", b"XY", "image/png")})
    resp = client.post(
        "/tools/_stub",
        files={"image": ("d.png", b"XY", "image/png")},
        data={"repeat": "2"},
    )
    assert resp.headers["x-cache"] == "MISS"
    assert len(CALLS) == 2


def test_validation_error_is_422(client):
    resp = client.post(
        "/tools/_stub",
        files={"image": ("d.png", b"XY", "image/png")},
        data={"repeat": "99"},  # exceeds le=8
    )
    assert resp.status_code == 422


def test_missing_file_is_422(client):
    assert client.post("/tools/_stub", data={"repeat": "1"}).status_code == 422


def test_upload_over_limit_is_413(client, monkeypatch):
    client.app.state.config.max_upload_mb = 0  # 0 MB → any upload is too large
    resp = client.post("/tools/_stub", files={"image": ("d.png", b"XY", "image/png")})
    assert resp.status_code == 413


def test_tool_exception_becomes_500(client, monkeypatch):
    def boom(self, req, ctx):
        raise RuntimeError("backbone exploded")

    monkeypatch.setattr(StubTool, "run", boom)
    resp = client.post("/tools/_stub", files={"image": ("d.png", b"ZZ", "image/png")})
    assert resp.status_code == 500
    assert "backbone exploded" in resp.json()["detail"]


def test_unknown_tool_is_404(client):
    assert client.post("/tools/nope", files={"image": ("d.png", b"X", "image/png")}).status_code == 404


# --- model warm-up ---------------------------------------------------------
# Regression: loaders used to be registered inside run(), so startup warm-up
# found nothing and every model loaded on the FIRST REQUEST instead. Unit tests
# passed; only a real deploy exposed it.

LOADED: list[str] = []


class WarmInput(BaseModel):
    image: FileBytes


class WarmTool(Tool):
    name = "_warm"
    summary = "declares model loaders"
    InputModel = WarmInput
    OutputModel = MediaResponse

    def model_loaders(self):
        return {
            "fake_a": lambda dev: LOADED.append("fake_a") or "A",
            "fake_b": lambda dev: LOADED.append("fake_b") or "B",
        }

    def run(self, req, ctx):
        return MediaResponse(
            data=ctx.models.get("fake_a").encode() + ctx.models.get("fake_b").encode(),
            media_type="application/octet-stream",
        )


@pytest.fixture
def warm_client(tmp_path, monkeypatch):
    monkeypatch.setenv("HVYM_DEVICE", "cpu")
    LOADED.clear()
    registry.unregister(WarmTool.name)
    registry.register(WarmTool)
    config = Config.from_env()
    config.cache_dir = tmp_path / "cache"
    config.workspace_dir = tmp_path / "work"
    with TestClient(create_app(config, discover=False)) as c:
        yield c
    registry.unregister(WarmTool.name)


def test_models_needed_defaults_to_declared_loaders():
    assert WarmTool().models_needed() == ["fake_a", "fake_b"]


def test_models_are_warmed_at_startup_not_first_request(warm_client):
    """The whole point of ModelCache: no request should pay the load cost."""
    body = warm_client.get("/healthz").json()
    assert sorted(body["models_loaded"]) == ["fake_a", "fake_b"]
    assert sorted(LOADED) == ["fake_a", "fake_b"]

    resp = warm_client.post("/tools/_warm", files={"image": ("a.png", b"x", "image/png")})
    assert resp.status_code == 200 and resp.content == b"AB"
    assert len(LOADED) == 2, "serving a request must not trigger another load"


def test_warm_up_disabled_leaves_models_lazy(tmp_path, monkeypatch):
    monkeypatch.setenv("HVYM_DEVICE", "cpu")
    LOADED.clear()
    registry.unregister(WarmTool.name)
    registry.register(WarmTool)
    config = Config.from_env()
    config.cache_dir = tmp_path / "c"
    config.workspace_dir = tmp_path / "w"
    config.warm_on_startup = False
    try:
        with TestClient(create_app(config, discover=False)) as c:
            assert c.get("/healthz").json()["models_loaded"] == []
            assert LOADED == []
            # still works, just lazily
            assert c.post("/tools/_warm", files={"image": ("a.png", b"x", "image/png")}).content == b"AB"
            assert sorted(LOADED) == ["fake_a", "fake_b"]
    finally:
        registry.unregister(WarmTool.name)
