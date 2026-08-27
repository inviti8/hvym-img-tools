"""Fetch model weights at build time, resiliently.

Baking weights is the point: a cold serverless worker must not download several
GB before serving its first job. But fetching them in CI hits a problem an
inline `snapshot_download` does not survive — **HuggingFace rate-limits
unauthenticated traffic, and GitHub Actions runners share IPs**, so a build can
fail with a 429 through no fault of its own.

So: retry with backoff, and use HF_TOKEN when one is available (HF grants
higher limits to authenticated requests). Usage:

    python fetch_weights.py microsoft/TRELLIS-image-large [more/repos ...]
"""
from __future__ import annotations

import os
import sys
import time

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

ATTEMPTS = int(os.environ.get("HF_FETCH_ATTEMPTS", "6"))
BASE_DELAY = float(os.environ.get("HF_FETCH_BASE_DELAY", "15"))


def fetch(repo_id: str) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    if token:
        print(f"  using an authenticated HF session for {repo_id}", flush=True)

    last: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            path = snapshot_download(repo_id, token=token)
            print(f"ok: {repo_id} -> {path}", flush=True)
            return path
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            # 429 is throttling and 5xx is the hub having a moment; both are
            # worth waiting out. Anything else (404, 401 on a gated repo) will
            # not improve with time, so fail immediately and say so.
            if status not in (429, 500, 502, 503, 504):
                raise
            last = exc
            delay = BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"  attempt {attempt}/{ATTEMPTS} for {repo_id} got HTTP {status}; "
                f"retrying in {delay:.0f}s",
                flush=True,
            )
            if attempt < ATTEMPTS:
                time.sleep(delay)
        except Exception as exc:  # noqa: BLE001 - transient network faults
            last = exc
            delay = BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"  attempt {attempt}/{ATTEMPTS} for {repo_id} failed "
                f"({type(exc).__name__}); retrying in {delay:.0f}s",
                flush=True,
            )
            if attempt < ATTEMPTS:
                time.sleep(delay)

    raise SystemExit(
        f"could not fetch {repo_id} after {ATTEMPTS} attempts: {last}\n"
        "If this is a 429, the build is being rate-limited as an anonymous "
        "client. Set an HF_TOKEN secret on the repository and pass it through "
        "as a build secret to raise the limit."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: fetch_weights.py <repo_id> [<repo_id> ...]")
    for repo in sys.argv[1:]:
        fetch(repo)
