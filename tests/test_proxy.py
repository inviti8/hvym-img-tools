"""Proxy tests — the RunPod upstream is mocked, so no network and no GPU.

The property that matters most: the proxy must present the *same* HTTP contract
as the direct server, so Inkternity's client is identical either way. And the
RunPod key must never appear in a response.
"""
from __future__ import annotations

import base64

import httpx
import pytest
from fastapi.testclient import TestClient

from hvym_img_tools import proxy as proxy_mod

GOOD = "proxy-test-key-long-enough-0001"
RUNPOD_KEY = "runpod-secret-must-never-leak-9999"
GLB = b"glTF" + b"\x00" * 64


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("HVYM_API_KEY", GOOD)
    monkeypatch.setenv("RUNPOD_API_KEY", RUNPOD_KEY)
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "ep123")
    monkeypatch.setenv("HVYM_DEVICE", "cpu")


class FakeUpstream:
    """Stands in for RunPod, capturing what the proxy actually sent."""

    def __init__(self, payload=None, status=200):
        self.payload = payload if payload is not None else {
            "status": "COMPLETED",
            "output": {
                "data": base64.b64encode(GLB).decode(),
                "media_type": "model/gltf-binary",
                "filename": "char.glb",
                "cached": False,
                "elapsed": 1.72,
                "tool_version": "0.1.0",
            },
        }
        self.status = status
        self.seen: dict = {}

    def install(self, monkeypatch, raises=None):
        upstream = self

        class _Client:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers=None, json=None):
                if raises is not None:
                    raise raises
                upstream.seen = {"url": url, "headers": headers or {}, "json": json or {}}
                return httpx.Response(
                    upstream.status,
                    json=upstream.payload,
                    request=httpx.Request("POST", url),
                )

        monkeypatch.setattr(proxy_mod.httpx, "AsyncClient", _Client)
        return upstream


def _client():
    return TestClient(proxy_mod.create_app())


def test_healthz_open_and_reports_config(env):
    body = _client().get("/healthz").json()
    assert body["status"] == "ok"
    assert body["mode"] == "proxy"
    assert body["auth"] is True
    assert body["runpod_configured"] is True


def test_healthz_never_exposes_the_runpod_key(env):
    text = _client().get("/healthz").text
    assert RUNPOD_KEY not in text


def test_requires_our_api_key(env, monkeypatch):
    FakeUpstream().install(monkeypatch)
    resp = _client().post("/tools/reangle", files={"image": ("a.png", b"x", "image/png")})
    assert resp.status_code == 401


def test_forwards_and_returns_binary(env, monkeypatch):
    up = FakeUpstream().install(monkeypatch)
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"RAW", "image/png")},
        data={"mc_resolution": "256"},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 200
    assert resp.content == GLB
    assert resp.headers["content-type"] == "model/gltf-binary"
    assert resp.headers["x-cache"] == "MISS"
    assert resp.headers["x-tool-version"] == "0.1.0"
    assert "char.glb" in resp.headers["content-disposition"]

    # what actually went upstream
    sent = up.seen["json"]["input"]
    assert sent["tool"] == "reangle"
    assert base64.b64decode(sent["image"]) == b"RAW"
    assert sent["mc_resolution"] == "256"
    assert up.seen["headers"]["Authorization"] == f"Bearer {RUNPOD_KEY}"
    assert up.seen["url"].endswith("/ep123/runsync")


def test_cache_hit_header_is_propagated(env, monkeypatch):
    FakeUpstream(payload={
        "status": "COMPLETED",
        "output": {"data": base64.b64encode(GLB).decode(), "cached": True,
                   "media_type": "model/gltf-binary"},
    }).install(monkeypatch)
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.headers["x-cache"] == "HIT"


def test_bearer_form_also_accepted(env, monkeypatch):
    FakeUpstream().install(monkeypatch)
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"Authorization": f"Bearer {GOOD}"},
    )
    assert resp.status_code == 200


def test_tool_error_becomes_500(env, monkeypatch):
    FakeUpstream(payload={"status": "COMPLETED",
                          "output": {"error": "backbone exploded"}}).install(monkeypatch)
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 500
    assert "backbone exploded" in resp.json()["detail"]


def test_failed_job_becomes_502(env, monkeypatch):
    FakeUpstream(payload={"status": "FAILED", "error": "worker died"}).install(monkeypatch)
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 502


def test_upstream_timeout_becomes_504(env, monkeypatch):
    FakeUpstream().install(monkeypatch, raises=httpx.TimeoutException("slow"))
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 504


def test_upstream_error_does_not_leak_key_or_url(env, monkeypatch):
    FakeUpstream().install(monkeypatch, raises=httpx.ConnectError("boom"))
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 502
    assert RUNPOD_KEY not in resp.text
    assert "api.runpod.ai" not in resp.text


def test_unconfigured_proxy_is_503(monkeypatch):
    monkeypatch.setenv("HVYM_API_KEY", GOOD)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
    monkeypatch.setenv("HVYM_DEVICE", "cpu")
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 503
