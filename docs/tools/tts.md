# `tts` — a line of dialog, spoken

Status: **live** on RunPod Serverless as endpoint `hvym-img-tts`
(`37q2tisf0fssjn`, EU-RO-1, 4090/L4, scale to zero, image `hvym-img-tts:0.1.0`).
Benchmarked 2026-10-18 (§5). Tool #3, and the first that isn't an image tool.

## 1. What and why

Keyternity's Dialog window sends a typed line and gets back the audio it lip-syncs.
Inkternity could use it as well.

The model is **Chatterbox** (Resemble AI): MIT code and weights, 23 languages, voice
cloning from a short reference clip, and an emotion "exaggeration" control. It's the
voice model that's impractical on an artist's machine (PyTorch, and best on a GPU),
so it runs here. Keyternity keeps Kokoro, which runs well on a CPU, local, and keeps
its local Chatterbox helper as the offline option.

Sovereign, as this repo requires: self-hosted weights, and no third-party voice
API. Keyternity dropped ElevenLabs for exactly this.

## 2. Contract

`POST /tools/tts`, multipart:

| field | | |
|---|---|---|
| `text` | required | The line, 1–600 characters. One line of dialog: Chatterbox stops after ~1000 speech tokens (~40 s), so clients split long passages. |
| `language` | `en` | `ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh`. English uses the English model; the rest use the multilingual one. |
| `exaggeration` | `0.5` | Emotion, 0.25–2: 0.5 is neutral, and higher is more expressive (and faster). |
| `cfg_weight` | `0.5` | Pacing guidance, 0–1; lower is slower and steadier. Try 0.3 with high exaggeration. |
| `temperature` | `0.8` | 0.05–2. |
| `seed` | `0` | Generation samples randomly, so the seed makes a take reproducible and cacheable. **A different seed is a different take.** |
| `reference` | none | **Optional file.** A clean 5–20 s clip of the voice to clone: WAV, FLAC, OGG or MP3, recognised by content. Anything else is a 422. Left out, Chatterbox's own voice speaks. |

**Response:** `200` with `audio/wav` (mono, 16-bit, 24 kHz). There are no word
timings; Keyternity's lip sync times clean generated speech from the audio itself.
Cached by `sha256` of every field, the reference's bytes included, so asking for the
same take twice is free.

`reference` is the repo's first **optional** file field. Core now supports
`FileBytes | None`: the server mounts it as an optional upload, `/tools` lists it
under `file_fields` but not `required_file_fields`, and the RunPod worker only
demands required files.

## 3. Consent and watermarking

- **Consent:** cloning someone's voice needs their consent. Clients gate the
  reference upload behind a checkbox (Keyternity does). The tool ships no voices of
  real people.
- **Watermarking:** every output carries Resemble's imperceptible **Perth
  watermark**. Chatterbox applies it itself; this tool doesn't add or remove
  anything.

## 4. Models and image

- **Models:**
  - English: `chatterbox.tts.ChatterboxTTS`, warmed at worker start.
  - The other languages: `chatterbox.mtl_tts.ChatterboxMultilingualTTS`, which
    loads on its first request, so a mostly-English worker holds one model.
  - Both come from the `ResembleAI/chatterbox` HuggingFace repo.
- **Image:** `docker/Dockerfile.tts` builds `hvym-img-tts`. It has its own image
  because it pins **torch 2.6.0 and transformers 5.2**, which the other images can't
  share.
  - Weights are baked in, and `HF_HUB_OFFLINE=1` keeps cold workers off the network.
  - The build proves both models load offline, and that the watermarker is ready.
- **GPU pools:** sm_90 and below (`ADA_24`, `AMPERE_24`, `AMPERE_16`). torch
  2.6.0+cu124 has no Blackwell kernels.

## 5. Measured on the live endpoint (2026-10-18)

`scripts/benchmark/tts_endpoint.py`, called directly on RunPod (not through the
proxy), so RunPod's own `delayTime` (queue and cold start) and `executionTime`
(work) are separate.

| job | delay | work | audio | work ÷ audio | cost* |
|---|---|---|---|---|---|
| first ever (image pull + load) | **578 s** | 1.9 s | 1.6 s | 1.15 | $0.0006 |
| warm, short English line | 0.13 s | 1.9 s | 1.8 s | 1.01–1.06 | $0.0006 |
| warm, long English line | 0.15 s | 5.2 s | 7.3 s | **0.71** | $0.0016 |
| emotional (exaggeration 1.2) | 0.13 s | 1.8 s | 1.8 s | 1.02 | $0.0006 |
| cloned voice, first line | 0.15 s | 5.1 s | 2.7 s | 1.91 | $0.0016 |
| same voice again | 0.14 s | 6.7 s | 8.5 s | 0.78 | $0.0021 |
| French, first (multilingual model loads) | 0.14 s | 13.7 s | 2.1 s | 6.56 | $0.0042 |
| French, warm | 0.13 s | 5.6 s | 6.9 s | 0.81 | $0.0017 |
| French, cloned | 0.15 s | 3.3 s | 2.9 s | 1.13 | $0.0010 |
| restart from idle, image cached | **0.87 s** | 1.9 s | | | |

\* The cost column is work × $0.00031/s. **The real spend for all eleven jobs,
including the first pull and the idle tails, was $0.0385** (the account balance
before and after).

- **Warm:** Chatterbox speaks at roughly real time on short lines and faster on
  long ones, at about **$0.0006–0.002 per line**.
- **Analysing a voice** costs about 3 s once. After that the voice is remembered
  (`VoiceMemory`), and the same voice's next line ran at 0.78× real time.
- **The multilingual model** costs about 8 s to load on its first non-English line,
  then it's as fast as English.
- **Cold start:**
  - The very first pull of the image took 9.6 minutes.
  - After that, RunPod parks an idle worker with the image cached. It isn't billed
    (the spend rate stayed at storage only), and it restarted in 0.87 s.
  - A worker RunPod has fully evicted will be in between (container start plus
    model load). It can't be forced on demand, so it wasn't measured.
- **Quality:**
  - Whisper (tiny.en) transcribed every English output exactly, cloned voices
    included.
  - Cloning carries the voice over. The reference (Windows' "David") has a median
    pitch of 101 Hz. Chatterbox's own voice is 143–162 Hz, but the cloned lines
    came out at 90–104 Hz, and French cloned at 111 Hz.
  - Listening is still the real test. The reference here was a synthetic voice, so
    repeat it with a real recording.

## 6. Not done yet

- **Route the proxy** (on the proxy's server): `sudo bash set_tool_endpoint.sh tts
  37q2tisf0fssjn`. The Keyternity client is built, and Inkternity can use the same
  call.
