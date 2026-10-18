"""Benchmark the `tts` endpoint on RunPod Serverless (docs/tools/tts.md §5).

Calls the endpoint directly (not through the proxy) so RunPod's own accounting
separates queue + cold start (`delayTime`) from work (`executionTime`):

  1. cold: the first line after the endpoint has been idle
  2. warm English: short and long lines, each a new seed so the cache can't answer
  3. cloning: a reference clip twice -- the second reuses the analysed voice
  4. French: the multilingual model loads lazily on its first line, then warm

Every result is saved as a WAV to listen to; the numbers go to bench.json.

    uv run python scripts/benchmark/tts_endpoint.py --endpoint <id> --reference voice.wav --out out/
Reads RUNPOD_API_KEY from the environment or .env.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

UA = "hvym-img-tools-bench/1.0"  # RunPod's firewall 403s urllib's default


def load_env() -> None:
    for path in [Path(__file__).resolve().parents[2] / ".env", Path.cwd() / ".env"]:
        if path.exists():
            for line in path.read_text().splitlines():
                k, _, v = line.strip().partition("=")
                if k and v and not k.startswith("#"):
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def call(url: str, key: str, body: dict | None = None, method: str = "POST") -> dict:
    req = urllib.request.Request(
        url, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": UA},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def job(endpoint: str, key: str, inp: dict) -> tuple[dict, float]:
    """Run one job to completion; (status json, wall seconds)."""
    base = f"https://api.runpod.ai/v2/{endpoint}"
    started = time.perf_counter()
    st = call(f"{base}/runsync", key, {"input": {"tool": "tts", **inp}})
    while st.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        time.sleep(2)
        st = call(f"{base}/status/{st['id']}", key, method="GET")
    return st, time.perf_counter() - started


def audio_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--reference", help="a clip to clone (5-20 s)")
    ap.add_argument("--out", default="tts_bench")
    ap.add_argument("--usd-per-s", type=float, default=0.00031,
                    help="GPU price per second, for the cost column (RunPod flex 24 GB tier)")
    args = ap.parse_args()
    key = os.environ["RUNPOD_API_KEY"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = base64.b64encode(Path(args.reference).read_bytes()).decode() if args.reference else None

    short = "Hello there. How are you today?"
    long = ("Well, I suppose that depends on what you mean by ready. The bags are packed, "
            "the car is outside, and the map is on the table, but my heart is still at home.")
    plan = [
        ("cold_en_short", {"text": short, "seed": 101}),
        ("warm_en_short_1", {"text": short, "seed": 102}),
        ("warm_en_short_2", {"text": short, "seed": 103}),
        ("warm_en_long", {"text": long, "seed": 104}),
        ("warm_en_emotional", {"text": short, "seed": 105, "exaggeration": 1.2, "cfg_weight": 0.3}),
    ]
    if ref:
        plan += [
            ("clone_first", {"text": short, "seed": 106, "reference": ref}),
            ("clone_again", {"text": long, "seed": 107, "reference": ref}),
        ]
    plan += [
        ("fr_first", {"text": "Bonjour, comment allez-vous aujourd'hui ?", "language": "fr", "seed": 108}),
        ("fr_warm", {"text": "Je suis prêt, mais mon cœur est encore à la maison.", "language": "fr", "seed": 109}),
    ]
    if ref:
        plan.append(("fr_clone", {"text": "Bonjour, je suis prêt.", "language": "fr", "seed": 110, "reference": ref}))

    rows = []
    for name, inp in plan:
        st, wall = job(args.endpoint, key, inp)
        row = {"name": name, "status": st.get("status"), "wall_s": round(wall, 2),
               "delay_s": round(st.get("delayTime", 0) / 1000, 2),
               "exec_s": round(st.get("executionTime", 0) / 1000, 2)}
        output = st.get("output") or {}
        if st.get("status") == "COMPLETED" and output.get("data"):
            data = base64.b64decode(output["data"])
            (out / f"{name}.wav").write_bytes(data)
            secs = audio_seconds(data)
            row.update(audio_s=round(secs, 2), cached=output.get("cached"),
                       rtf=round(row["exec_s"] / max(secs, 1e-3), 2),
                       usd=round(row["exec_s"] * args.usd_per_s, 5))
        else:
            row["error"] = output.get("error") or st.get("error") or st
        rows.append(row)
        print(json.dumps(row), flush=True)

    (out / "bench.json").write_text(json.dumps(rows, indent=1))
    print(f"\n{len(rows)} jobs; WAVs and bench.json in {out}")


if __name__ == "__main__":
    main()
