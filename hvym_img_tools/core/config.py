"""Process-wide configuration, environment-driven.

Deliberately plain: a serverless container is configured by env vars, not files.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw).expanduser() if raw else default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Config:
    """Runtime knobs. `Config.from_env()` is the only intended constructor."""

    #: Where cached results live. Persist this across restarts to keep the cache warm.
    cache_dir: Path
    #: Scratch space for a single request.
    workspace_dir: Path
    #: Where model weights are looked up (bake these into the image for fast cold start).
    models_dir: Path
    #: Load every tool's models at startup rather than lazily on first request.
    warm_on_startup: bool = True
    #: Device hint passed to tools. "auto" resolves at load time, never at import.
    device: str = "auto"
    #: Upper bound on a single upload, in megabytes.
    max_upload_mb: int = 32
    #: Accepted API keys. Empty = auth disabled (dev only). Comma-separated in
    #: HVYM_API_KEY so a key can be rotated without downtime.
    api_keys: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Config":
        tmp = Path(tempfile.gettempdir())
        return cls(
            cache_dir=_env_path("HVYM_CACHE_DIR", tmp / "hvym-img-tools" / "cache"),
            workspace_dir=_env_path("HVYM_WORKSPACE_DIR", tmp / "hvym-img-tools" / "work"),
            models_dir=_env_path("HVYM_MODELS_DIR", Path("/workspace/models")),
            warm_on_startup=_env_bool("HVYM_WARM_ON_STARTUP", True),
            device=os.environ.get("HVYM_DEVICE", "auto"),
            max_upload_mb=int(os.environ.get("HVYM_MAX_UPLOAD_MB", "32")),
            api_keys=tuple(
                k.strip() for k in os.environ.get("HVYM_API_KEY", "").split(",") if k.strip()
            ),
        )

    def resolve_device(self) -> str:
        """Resolve "auto" to a concrete device. Imports torch lazily, by design."""
        if self.device != "auto":
            return self.device
        try:
            import torch  # noqa: PLC0415 - intentionally lazy, keeps core CPU-importable
        except ImportError:
            return "cpu"
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def ensure_dirs(self) -> None:
        for path in (self.cache_dir, self.workspace_dir):
            path.mkdir(parents=True, exist_ok=True)
