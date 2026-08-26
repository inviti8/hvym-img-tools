"""API-key auth tests.

Scope note: these verify the mechanism does what it claims. They do NOT claim the
scheme is strong — a key shipped in a desktop binary is extractable by design
(see hvym_img_tools/core/auth.py and docs/AUTH.md).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from hvym_img_tools.core import registry
from hvym_img_tools.core.auth import ApiKeyAuth, extract_key, generate_key
from hvym_img_tools.core.config import Config
from hvym_img_tools.core.server import create_app
from hvym_img_tools.core.tool import Context, FileBytes, MediaResponse, Tool

GOOD = "test-key-that-is-long-enough-01"
ALSO_GOOD = "rotation-key-long-enough-000002"


class AuthInput(BaseModel):
    image: FileBytes


class AuthTool(Tool):
    name = "_auth"
    summary = "auth test tool"
    InputModel = AuthInput
    OutputModel = MediaResponse

    def run(self, req: AuthInput, ctx: Context) -> MediaResponse:
        return MediaResponse(data=b"ok", media_type="application/octet-stream")


def _client(tmp_path, keys: tuple[str, ...]):
    registry.unregister(AuthTool.name)
    registry.register(AuthTool)
    config = Config.from_env()
    config.device = "cpu"
    config.cache_dir = tmp_path / "c"
    config.workspace_dir = tmp_path / "w"
    config.api_keys = keys
    return TestClient(create_app(config, discover=False))


@pytest.fixture
def secured(tmp_path):
    with _client(tmp_path, (GOOD, ALSO_GOOD)) as c:
        yield c
    registry.unregister(AuthTool.name)


@pytest.fixture
def open_server(tmp_path):
    with _client(tmp_path, ()) as c:
        yield c
    registry.unregister(AuthTool.name)


# --- the unit ---------------------------------------------------------------

def test_disabled_auth_allows_everything():
    assert ApiKeyAuth.from_keys(()).verify(None) is True


def test_verify_accepts_only_exact_keys():
    auth = ApiKeyAuth.from_keys((GOOD,))
    assert auth.verify(GOOD) is True
    assert auth.verify(GOOD + "x") is False
    assert auth.verify(GOOD[:-1]) is False
    assert auth.verify("") is False
    assert auth.verify(None) is False


def test_multiple_keys_support_rotation():
    auth = ApiKeyAuth.from_keys((GOOD, ALSO_GOOD))
    assert auth.verify(GOOD) and auth.verify(ALSO_GOOD)


def test_blank_keys_are_dropped_not_treated_as_valid():
    """A stray comma in HVYM_API_KEY must not create an empty accepted key."""
    auth = ApiKeyAuth.from_keys(("", "   ", GOOD))
    assert auth.keys == frozenset({GOOD})
    assert auth.verify("") is False


def test_extract_key_from_either_header():
    assert extract_key("abc", None) == "abc"
    assert extract_key(None, "Bearer abc") == "abc"
    assert extract_key(None, "bearer abc") == "abc"
    assert extract_key(None, "Basic abc") is None
    assert extract_key(None, None) is None


def test_generated_key_is_strong():
    key = generate_key()
    assert len(key) >= 32
    assert generate_key() != key


# --- the wiring -------------------------------------------------------------

def test_tool_requires_key(secured):
    resp = secured.post("/tools/_auth", files={"image": ("a.png", b"x", "image/png")})
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


def test_tool_accepts_api_key_header(secured):
    resp = secured.post(
        "/tools/_auth",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 200 and resp.content == b"ok"


def test_tool_accepts_bearer(secured):
    resp = secured.post(
        "/tools/_auth",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"Authorization": f"Bearer {ALSO_GOOD}"},
    )
    assert resp.status_code == 200


def test_wrong_key_rejected(secured):
    resp = secured.post(
        "/tools/_auth",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": "nope"},
    )
    assert resp.status_code == 401


def test_error_does_not_leak_why(secured):
    """Same body for a missing key and a wrong key — no oracle for probing."""
    missing = secured.post("/tools/_auth", files={"image": ("a.png", b"x", "image/png")})
    wrong = secured.post(
        "/tools/_auth",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": "nope"},
    )
    assert missing.json() == wrong.json()


def test_tools_listing_is_protected(secured):
    assert secured.get("/tools").status_code == 401
    assert secured.get("/tools", headers={"X-API-Key": GOOD}).status_code == 200


def test_healthz_stays_open_for_probes(secured):
    """Orchestrator health probes carry no credentials; healthz must not 401."""
    resp = secured.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["auth"] is True


def test_open_server_allows_calls_and_reports_auth_off(open_server):
    assert open_server.get("/healthz").json()["auth"] is False
    resp = open_server.post("/tools/_auth", files={"image": ("a.png", b"x", "image/png")})
    assert resp.status_code == 200
