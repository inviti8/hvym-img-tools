"""Core framework: the Tool contract, registry, model cache, result cache, image utils.

Import-light and CPU-only by design (AGENTS.md §6) — `server` pulls FastAPI, and
model loaders pull torch, but only when actually used.
"""
from .cache import CacheEntry, ResultCache, hash_parts
from .config import Config
from .models import ModelCache
from .registry import discover, get, names, register
from .tool import Context, FileBytes, MediaResponse, Tool

__all__ = [
    "CacheEntry", "ResultCache", "hash_parts",
    "Config", "ModelCache",
    "discover", "get", "names", "register",
    "Context", "FileBytes", "MediaResponse", "Tool",
]
