# `tts` — a line of dialog, spoken

Status: **built, not yet deployed or benchmarked** (§5). Tool #3, and the first
that isn't an image tool.

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

## 5. Not done yet

- **Benchmark (AGENTS.md §0):**
  - Cold start, and warm seconds per second of audio, on a 4090 or L4.
  - Cost per line.
  - With and without a reference.

  Record it in `docs/BENCHMARK.md` before relying on the numbers.
- **Deploy:** build (`gh workflow run images.yml --ref <branch> -f tag=0.1.0 -f
  only=tts`), then `scripts/deploy_runpod.py --image ghcr.io/inviti8/hvym-img-tts:0.1.0
  --name hvym-img-tts`, then route the proxy with `set_tool_endpoint.sh tts
  <endpoint-id>`.
- **The Keyternity client:** a Hosted Chatterbox voice in the Dialog window. Its
  endpoint and key go in Settings → Voices, with Inkternity's timeouts and error
  handling (docs/CLIENT.md).
