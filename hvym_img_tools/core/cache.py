"""Result cache keyed by `sha256(input + params)` (AGENTS.md §4, §8).

Reangle's reconstruction is the case this exists for: one request per drawing,
then every re-request is instant.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def hash_parts(parts: Iterable[bytes]) -> str:
    """Stable sha256 over length-prefixed parts.

    Length prefixing matters: without it (b"ab", b"c") and (b"a", b"bc") collide.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


@dataclass(slots=True)
class CacheEntry:
    key: str
    path: Path
    media_type: str

    def read(self) -> bytes:
        return self.path.read_bytes()


class ResultCache:
    """Content-addressed store on disk. Safe for concurrent writers."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        shard = self.root / key[:2]
        return shard / key, shard / f"{key}.type"

    def get(self, key: str) -> CacheEntry | None:
        blob, meta = self._paths(key)
        if not blob.exists():
            return None
        media_type = meta.read_text().strip() if meta.exists() else "application/octet-stream"
        return CacheEntry(key=key, path=blob, media_type=media_type)

    def put(self, key: str, data: bytes, media_type: str = "application/octet-stream") -> CacheEntry:
        blob, meta = self._paths(key)
        blob.parent.mkdir(parents=True, exist_ok=True)
        # Write to a unique temp file then atomically rename, so a concurrent
        # reader never observes a half-written blob.
        tmp = blob.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, blob)
        meta.write_text(media_type)
        return CacheEntry(key=key, path=blob, media_type=media_type)

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def stats(self) -> dict[str, int]:
        blobs = [p for p in self.root.rglob("*") if p.is_file() and p.suffix != ".type"]
        return {"entries": len(blobs), "bytes": sum(p.stat().st_size for p in blobs)}
