"""Operator switch: keep a Serverless worker warm for a demo.

    uv run python scripts/warm.py on       # start a worker and keep it up
    uv run python scripts/warm.py wait     # block until it can serve (~1-4 min)
    uv run python scripts/warm.py status   # is it up, how long, what has it cost
    uv run python scripts/warm.py off      # release it

**This is the demo switch, not the product design.** It is deliberately a plain
switch with no auto-expiry: you turn it on, it stays on until you turn it off.
That is the right semantic for a tool an operator drives by hand, and it is the
wrong one for anything a client holds -- see docs/WARMING.md.

It costs real money while on (~$1.10/hour, roughly 2,000 images' worth of
compute per idle hour), so `status` reports elapsed time and estimated spend,
and `on` tells you what you just started paying for.

Implemented by setting the endpoint's workersMin to 1 rather than by pinging it:
a switch should not need a process babysitting it, and RunPod's own config is
the one piece of state guaranteed to survive this script exiting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REST = "https://rest.runpod.io/v1"
QUEUE = "https://api.runpod.ai/v2"

#: 24GB flex tier, approximate. Verify against `/billing/endpoints` once a day of
#: real usage has posted -- see docs/WARMING.md.
USD_PER_SECOND = 0.00031

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".warm_state.json"


def load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def call(url: str, key: str, method: str = "GET", body: dict | None = None) -> object:
    req = urllib.request.Request(
        url, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"RunPod {method} failed ({exc.code}): {exc.read().decode()[:400]}") from exc


def human(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def read_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["on", "off", "status", "wait"])
    ap.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID", ""))
    ap.add_argument("--timeout", type=int, default=420,
                    help="seconds for `wait` to give up (cold start is up to ~260s)")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set (put it in .env)")
    if not args.endpoint_id:
        raise SystemExit("RUNPOD_ENDPOINT_ID is not set (put it in .env, or pass --endpoint-id)")

    ep_url = f"{REST}/endpoints/{args.endpoint_id}"
    health_url = f"{QUEUE}/{args.endpoint_id}/health"

    def health() -> dict:
        return (call(health_url, key) or {}).get("workers", {}) or {}

    def is_on() -> bool:
        return int((call(ep_url, key) or {}).get("workersMin") or 0) > 0

    if args.action == "on":
        if is_on():
            print("already on -- `status` for elapsed and spend")
            return 0
        call(f"{ep_url}", key, "PATCH", {"workersMin": 1})
        STATE.write_text(json.dumps({"since": time.time(), "endpoint": args.endpoint_id}))
        print(f"warm ON  ({args.endpoint_id})")
        print(f"  billing now, ~${USD_PER_SECOND * 3600:.2f}/hour until you run `warm.py off`")
        print("  first worker takes up to ~4 min to pull the image; `warm.py wait` blocks until ready")
        return 0

    if args.action == "off":
        state = read_state()
        call(f"{ep_url}", key, "PATCH", {"workersMin": 0})
        STATE.unlink(missing_ok=True)
        print(f"warm OFF ({args.endpoint_id})")
        if state.get("since"):
            elapsed = time.time() - state["since"]
            print(f"  was on for {human(elapsed)}, roughly ${elapsed * USD_PER_SECOND:.2f}")
        print("  the worker drops after its idle timeout; scale-to-zero resumes")
        return 0

    if args.action == "status":
        on = is_on()
        w = health()
        ready = int(w.get("ready", 0))
        print(f"switch:  {'ON' if on else 'off'}")
        print(f"workers: ready={ready} initializing={w.get('initializing', 0)} "
              f"idle={w.get('idle', 0)} running={w.get('running', 0)}")
        if ready:
            print("serving: warm -- a request now skips the cold start")
        elif on:
            print("serving: still starting; `wait` blocks until ready")
        else:
            print("serving: cold -- next request pays up to ~260s")
        state = read_state()
        if on and state.get("since"):
            elapsed = time.time() - state["since"]
            print(f"elapsed: {human(elapsed)}  (~${elapsed * USD_PER_SECOND:.2f} so far)")
        elif on:
            print("elapsed: unknown (turned on elsewhere)")
        return 0

    # wait
    if not is_on():
        print("switch is off -- run `warm.py on` first", file=sys.stderr)
        return 1
    started = time.time()
    while time.time() - started < args.timeout:
        w = health()
        if int(w.get("ready", 0)) > 0:
            print(f"\nready after {human(time.time() - started)}")
            return 0
        print(f"\r  warming... {human(time.time() - started)} "
              f"(initializing={w.get('initializing', 0)})   ", end="", flush=True)
        time.sleep(5)
    print(f"\nstill not ready after {human(args.timeout)}; `status` for detail", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
