"""`TtsTool` — a line of dialog spoken by Chatterbox (docs/tools/tts.md).

Tool #3, and the first that isn't an image tool: Keyternity's Dialog window sends
a typed line and gets back the audio it lip-syncs. Chatterbox (Resemble AI, MIT
code and weights) speaks 23 languages, clones a voice from a short reference clip,
and has an emotion "exaggeration" control. It's the voice model that is impractical
on an artist's machine (PyTorch and a GPU), which is why it lives here; Keyternity
keeps Kokoro, which runs well on a CPU, local.

Every output carries Resemble's imperceptible Perth watermark (Chatterbox applies
it itself). Cloning someone's voice needs their consent; clients gate the reference
upload behind a consent checkbox, and this tool never ships voices of real people.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ...core.tool import Context, FileBytes, MediaResponse, Tool
from .audio import sniff_ext, to_wav

#: ModelCache keys. English has its own (better) model; the multilingual one covers
#: the other 22 languages.
CHATTERBOX = "chatterbox"
CHATTERBOX_MTL = "chatterbox_mtl"

#: Chatterbox Multilingual's languages (chatterbox.mtl_tts.SUPPORTED_LANGUAGES, 0.1.7).
LANGUAGES = (
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
)


def load_chatterbox(device: str) -> Any:
    from chatterbox.tts import ChatterboxTTS

    return ChatterboxTTS.from_pretrained(device=device)


def load_chatterbox_mtl(device: str) -> Any:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS.from_pretrained(device=device)


class TtsInput(BaseModel):
    """Multipart: the line and how to say it, plus an optional voice to clone."""

    text: str = Field(
        min_length=1,
        max_length=600,
        description=(
            "The line, punctuation and all. One line of dialog: Chatterbox stops "
            "after ~1000 speech tokens (about 40 s), so long passages are split "
            "by the client."
        ),
    )
    language: str = Field(default="en", description=f"ISO 639-1: {', '.join(LANGUAGES)}")
    exaggeration: float = Field(
        default=0.5, ge=0.25, le=2.0,
        description="Emotion: 0.5 is neutral; higher is more expressive (and faster).",
    )
    cfg_weight: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Pacing guidance; lower is slower and steadier (try 0.3 with high exaggeration).",
    )
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)
    seed: int = Field(
        default=0, ge=0, le=2**31 - 1,
        description=(
            "Generation samples randomly, so the seed makes a take reproducible and "
            "cacheable. A different seed is a different take."
        ),
    )
    reference: FileBytes | None = Field(
        default=None,
        description=(
            "A clean clip of the voice to clone (WAV, FLAC, OGG or MP3; 5-20 s). "
            "Only with the voice owner's consent. Left out: Chatterbox's own voice."
        ),
    )

    @field_validator("reference")
    @classmethod
    def _audio_reference(cls, v: bytes | None) -> bytes | None:
        if v is not None:
            sniff_ext(v)  # a 422 here, not a failure inside the model
        return v

    @field_validator("language")
    @classmethod
    def _known_language(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in LANGUAGES:
            raise ValueError(f"unsupported language {v!r}; one of {', '.join(LANGUAGES)}")
        return v


def synthesize(model: Any, req: TtsInput, reference: Path | None) -> tuple[Any, int]:
    """Run Chatterbox: (waveform, sample rate). Seeded for reproducible takes."""
    import numpy as np

    random.seed(req.seed)
    np.random.seed(req.seed)
    try:
        import torch

        torch.manual_seed(req.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(req.seed)
    except ImportError:  # the CPU test environment has no torch
        pass

    kwargs: dict[str, Any] = {
        "audio_prompt_path": str(reference) if reference else None,
        "exaggeration": req.exaggeration,
        "cfg_weight": req.cfg_weight,
        "temperature": req.temperature,
    }
    if req.language != "en":
        kwargs["language_id"] = req.language
    wav = model.generate(req.text, **kwargs)
    # a tensor [1, N] from Chatterbox; anything array-like from a stand-in
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return wav, int(model.sr)


class TtsTool(Tool):
    name = "tts"
    summary = "Speak a line of dialog (Chatterbox: 23 languages, voice cloning, emotion)"
    version = "0.1.0"
    InputModel = TtsInput
    OutputModel = MediaResponse

    def model_loaders(self) -> dict[str, Any]:
        return {CHATTERBOX: load_chatterbox, CHATTERBOX_MTL: load_chatterbox_mtl}

    def models_needed(self) -> list[str]:
        # English warm; the multilingual model loads on its first request, so a
        # worker serving mostly English doesn't hold a second model in memory.
        return [CHATTERBOX]

    def warmup(self, ctx: Context) -> None:
        # The first generate compiles kernels; pay it at worker start, not on a line.
        synthesize(ctx.models.get(CHATTERBOX), TtsInput(text="Ready."), None)

    def run(self, req: TtsInput, ctx: Context) -> MediaResponse:
        model = ctx.models.get(CHATTERBOX if req.language == "en" else CHATTERBOX_MTL)
        reference: Path | None = None
        try:
            if req.reference:
                ctx.workspace.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=ctx.workspace, suffix=sniff_ext(req.reference), delete=False
                ) as f:
                    f.write(req.reference)
                    reference = Path(f.name)
            wav, rate = synthesize(model, req, reference)
        finally:
            if reference is not None:
                reference.unlink(missing_ok=True)
        return MediaResponse(data=to_wav(wav, rate), media_type="audio/wav", filename="line.wav")
