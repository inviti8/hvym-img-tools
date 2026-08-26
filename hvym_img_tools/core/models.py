"""`ModelCache` — load heavy models once, share them across tools and requests.

Tools declare model keys in `models_needed()`; loaders are registered by whoever
owns the weights (a tool package, at import time). Nothing here imports torch:
loaders do that lazily, so `core` stays CPU-importable (AGENTS.md §6).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

log = logging.getLogger(__name__)

#: A loader takes the resolved device and returns a ready-to-use model handle.
Loader = Callable[[str], Any]


class ModelCache:
    """Thread-safe, lazily-populated registry of warm models."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._loaders: dict[str, Loader] = {}
        self._models: dict[str, Any] = {}
        self._load_times: dict[str, float] = {}
        # Per-key locks so warming model A never blocks a request needing model B.
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def register(self, key: str, loader: Loader, *, replace: bool = False) -> None:
        with self._guard:
            if key in self._loaders and not replace:
                raise ValueError(f"model loader {key!r} already registered")
            self._loaders[key] = loader
            self._locks.setdefault(key, threading.Lock())

    def registered(self) -> list[str]:
        return sorted(self._loaders)

    def loaded(self) -> list[str]:
        return sorted(self._models)

    def _lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def get(self, key: str) -> Any:
        """Return the model, loading it on first use."""
        model = self._models.get(key)
        if model is not None:
            return model
        if key not in self._loaders:
            raise KeyError(
                f"no loader registered for model {key!r}; registered: {self.registered()}"
            )
        with self._lock_for(key):
            if key in self._models:  # another thread won the race
                return self._models[key]
            started = time.perf_counter()
            log.info("loading model %s on %s", key, self.device)
            model = self._loaders[key](self.device)
            elapsed = time.perf_counter() - started
            self._models[key] = model
            self._load_times[key] = elapsed
            log.info("loaded model %s in %.2fs", key, elapsed)
            return model

    def warm(self, keys: list[str]) -> dict[str, float]:
        """Preload the given keys. Returns per-key load seconds."""
        for key in keys:
            self.get(key)
        return {k: self._load_times.get(k, 0.0) for k in keys}

    def load_times(self) -> dict[str, float]:
        return dict(self._load_times)

    def evict(self, key: str) -> bool:
        with self._lock_for(key):
            existed = self._models.pop(key, None) is not None
            self._load_times.pop(key, None)
        return existed
