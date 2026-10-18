"""`tts` tool tests — no GPU, no Chatterbox.

The model is faked, so these cover what we wrote: the contract (an optional
voice reference, languages, bounds), which model a language uses, how the
reference reaches the model, seeding, the WAV encoding, the cache key, and the
whole HTTP path with and without the optional upload.
"""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hvym_img_tools.core import registry
from hvym_img_tools.core.cache import ResultCache
from hvym_img_tools.core.config import Config
from hvym_img_tools.core.models import ModelCache
from hvym_img_tools.core.server import create_app
from hvym_img_tools.core.tool import Context
from hvym_img_tools.tools.tts.audio import sniff_ext, to_wav
from hvym_img_tools.tools.tts.tool import CHATTERBOX, CHATTERBOX_MTL, TtsInput, TtsTool


class FakeChatterbox:
    """Records what it was asked; 'speaks' a short tone at Chatterbox's 24 kHz."""

    sr = 24_000

    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict] = []

    def generate(self, text, **kw):
        prompt = kw.get("audio_prompt_path")
        self.calls.append({
            "text": text,
            **kw,
            "prompt_bytes": open(prompt, "rb").read() if prompt else None,
            "draw": float(np.random.random()),  # shows the seed took effect
        })
        t = np.arange(self.sr // 4) / self.sr
        return (0.5 * np.sin(2 * np.pi * 220 * t))[None, :]


def _wav(seconds: float = 0.1, rate: int = 16_000) -> bytes:
    return to_wav(np.zeros(int(seconds * rate)), rate)


@pytest.fixture
def ctx(tmp_path):
    models = ModelCache("cpu")
    en, mtl = FakeChatterbox("en"), FakeChatterbox("mtl")
    models.register(CHATTERBOX, lambda device: en)
    models.register(CHATTERBOX_MTL, lambda device: mtl)
    c = Context(models=models, cache=ResultCache(tmp_path / "cache"), workspace=tmp_path / "work", config=Config.from_env())
    c.extras.update(en=en, mtl=mtl)
    return c


# --- contract --------------------------------------------------------------

def test_registered_alongside_the_image_tools():
    registry.discover()
    assert {"tts", "mesh", "reangle"} <= set(registry.names())


def test_reference_is_an_optional_upload():
    assert TtsTool.file_fields() == ("reference",)
    assert TtsTool.required_file_fields() == ()
    assert TtsInput(text="Hi").reference is None


def test_languages_and_bounds():
    assert TtsInput(text="Hallo", language="DE").language == "de"
    for bad in [{"language": "xx"}, {"exaggeration": 3}, {"cfg_weight": -0.1}, {"seed": -1}]:
        with pytest.raises(ValueError):
            TtsInput(text="Hi", **bad)
    with pytest.raises(ValueError):
        TtsInput(text="")
    with pytest.raises(ValueError):
        TtsInput(text="x" * 601)


def test_reference_must_be_audio():
    with pytest.raises(ValueError, match="WAV, FLAC, OGG or MP3"):
        TtsInput(text="Hi", reference=b"not audio at all")
    assert TtsInput(text="Hi", reference=_wav()).reference


def test_only_english_is_warmed():
    t = TtsTool()
    assert sorted(t.model_loaders()) == [CHATTERBOX, CHATTERBOX_MTL]
    assert t.models_needed() == [CHATTERBOX]


# --- running ---------------------------------------------------------------

def test_english_uses_the_english_model_and_returns_wav(ctx):
    out = TtsTool().run(TtsInput(text="Hello there.", exaggeration=0.7), ctx)
    assert out.media_type == "audio/wav"
    with wave.open(io.BytesIO(out.data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24_000)
        assert w.getnframes() == 6_000
    call = ctx.extras["en"].calls[0]
    assert call["text"] == "Hello there." and call["exaggeration"] == 0.7
    assert "language_id" not in call, "the English model takes no language"
    assert call["audio_prompt_path"] is None
    assert not ctx.extras["mtl"].calls


def test_other_languages_use_the_multilingual_model(ctx):
    TtsTool().run(TtsInput(text="Bonjour.", language="fr"), ctx)
    call = ctx.extras["mtl"].calls[0]
    assert call["language_id"] == "fr"
    assert not ctx.extras["en"].calls


def test_the_reference_reaches_the_model_as_a_named_file_then_goes(ctx):
    ref = _wav()
    TtsTool().run(TtsInput(text="Hi.", reference=ref), ctx)
    call = ctx.extras["en"].calls[0]
    assert call["audio_prompt_path"].endswith(".wav")
    assert call["prompt_bytes"] == ref
    assert not list(ctx.workspace.glob("*")), "the reference is deleted after use"


def test_seed_makes_takes_reproducible(ctx):
    tool = TtsTool()
    for seed in (1, 1, 2):
        tool.run(TtsInput(text="Hi.", seed=seed), ctx)
    draws = [c["draw"] for c in ctx.extras["en"].calls]
    assert draws[0] == draws[1] != draws[2]


def test_seed_and_reference_change_the_cache_key():
    from hvym_img_tools.core.cache import hash_parts

    key = lambda **kw: hash_parts(TtsTool().cache_key_parts(TtsInput(text="Hi.", **kw)))
    assert key() == key()
    assert key() != key(seed=1)
    assert key() != key(reference=_wav())
    assert key(reference=_wav(0.1)) != key(reference=_wav(0.2))


def test_sniffing_audio():
    assert sniff_ext(_wav()) == ".wav"
    assert sniff_ext(b"fLaC....") == ".flac"
    assert sniff_ext(b"OggS....") == ".ogg"
    assert sniff_ext(b"ID3\x04...") == ".mp3"
    assert sniff_ext(bytes([0xFF, 0xFB, 0x90, 0x00])) == ".mp3"


# --- over HTTP ---------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HVYM_DEVICE", "cpu")
    monkeypatch.setenv("HVYM_WARM_ON_STARTUP", "0")
    fakes = {CHATTERBOX: FakeChatterbox("en"), CHATTERBOX_MTL: FakeChatterbox("mtl")}

    class FakeTts(TtsTool):
        def model_loaders(self):
            return {k: (lambda device, m=m: m) for k, m in fakes.items()}

    registry.register(FakeTts, replace=True)
    config = Config.from_env()
    config.cache_dir = tmp_path / "cache"
    config.workspace_dir = tmp_path / "work"
    with TestClient(create_app(config, discover=False)) as c:
        c.fakes = fakes
        yield c
    registry.register(TtsTool, replace=True)


def test_http_without_a_reference(client):
    r = client.post("/tools/tts", data={"text": "Hello there.", "seed": "3"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    # the same take again comes from the cache
    r2 = client.post("/tools/tts", data={"text": "Hello there.", "seed": "3"})
    assert r2.headers["X-Cache"] == "HIT"
    assert len(client.fakes[CHATTERBOX].calls) == 1


def test_http_with_a_reference(client):
    ref = _wav()
    r = client.post("/tools/tts", data={"text": "Hi."}, files={"reference": ("me.wav", ref, "audio/wav")})
    assert r.status_code == 200, r.text
    assert client.fakes[CHATTERBOX].calls[0]["prompt_bytes"] == ref


def test_http_rejects_a_bad_reference(client):
    r = client.post("/tools/tts", data={"text": "Hi."}, files={"reference": ("x.wav", b"nope", "audio/wav")})
    assert r.status_code == 422


def test_listed_with_an_optional_file(client):
    entry = next(t for t in client.get("/tools").json()["tools"] if t["name"] == "tts")
    assert entry["file_fields"] == ["reference"] and entry["required_file_fields"] == []


# --- the RunPod worker --------------------------------------------------------

def test_worker_takes_a_job_with_or_without_the_optional_reference(ctx, monkeypatch):
    import base64

    from hvym_img_tools import serverless

    ctx.config.cache_dir = ctx.workspace.parent / "wcache"
    ctx.cache = ResultCache(ctx.config.cache_dir)
    monkeypatch.setattr(serverless, "_CTX", ctx)
    monkeypatch.setattr(serverless, "_TOOLS", {"tts": TtsTool()})
    out = serverless.handler({"input": {"tool": "tts", "text": "Hello.", "seed": 2}})
    assert "error" not in out, out
    assert base64.b64decode(out["data"])[:4] == b"RIFF"
    ref = base64.b64encode(_wav()).decode()
    out = serverless.handler({"input": {"tool": "tts", "text": "Hello.", "reference": ref}})
    assert "error" not in out, out
    assert ctx.extras["en"].calls[-1]["prompt_bytes"] == _wav()
