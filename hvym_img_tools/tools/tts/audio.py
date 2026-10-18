"""Audio helpers for `tts`: plain stdlib + numpy, so they test on CPU."""
from __future__ import annotations

import io
import wave

import numpy as np


def sniff_ext(data: bytes) -> str:
    """The file extension a voice reference's bytes are, by their magic numbers.

    Chatterbox loads the reference through librosa, which picks a decoder by
    extension; the upload arrives without a filename, so it's named here.
    """
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"OggS":
        return ".ogg"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return ".mp3"
    raise ValueError("the voice reference isn't a WAV, FLAC, OGG or MP3 file")


def to_wav(samples: np.ndarray, rate: int) -> bytes:
    """Mono float samples (-1..1) as 16-bit PCM WAV bytes."""
    s = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = (np.clip(s, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
