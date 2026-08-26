"""Tool registry — discovery and mounting.

Adding a tool means adding a package under `tools/` and calling `register()`.
No edits here, none to sibling tools (AGENTS.md §2, §7).

Mounting is split out of `server.py` so the registry itself stays importable
without FastAPI installed — `core` must work for tests and CLI use with only the
base dependencies.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Iterator

from .tool import Tool

log = logging.getLogger(__name__)

_REGISTRY: dict[str, type[Tool]] = {}


def register(tool_cls: type[Tool], *, replace: bool = False) -> type[Tool]:
    """Register a tool class. Usable as a decorator."""
    if not isinstance(tool_cls, type) or not issubclass(tool_cls, Tool):
        raise TypeError(f"expected a Tool subclass, got {tool_cls!r}")
    for attr in ("name", "summary", "InputModel", "OutputModel"):
        if not getattr(tool_cls, attr, None):
            raise TypeError(f"{tool_cls.__name__} is missing required attribute {attr!r}")
    name = tool_cls.name
    if name in _REGISTRY and not replace and _REGISTRY[name] is not tool_cls:
        raise ValueError(
            f"tool {name!r} already registered by {_REGISTRY[name].__module__}; "
            "names must be unique"
        )
    _REGISTRY[name] = tool_cls
    log.debug("registered tool %s (%s)", name, tool_cls.__module__)
    return tool_cls


def unregister(name: str) -> bool:
    return _REGISTRY.pop(name, None) is not None


def get(name: str) -> type[Tool]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown tool {name!r}; available: {names()}") from None


def names() -> list[str]:
    return sorted(_REGISTRY)


def all_tools() -> Iterator[type[Tool]]:
    for name in names():
        yield _REGISTRY[name]


def describe_all() -> list[dict]:
    return [t.describe() for t in all_tools()]


def discover(package: str = "hvym_img_tools.tools") -> list[str]:
    """Import every subpackage of `package` so its `register()` calls run.

    A tool that fails to import is logged and skipped rather than taking the
    whole server down — one broken tool must not deny service to the others.
    """
    try:
        pkg = importlib.import_module(package)
    except ModuleNotFoundError:
        log.warning("tool package %s not found", package)
        return []

    found: list[str] = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        module = f"{package}.{info.name}"
        try:
            importlib.import_module(module)
            found.append(module)
        except Exception:  # noqa: BLE001 - isolate a broken tool
            log.exception("failed to import tool module %s; skipping", module)
    return found
