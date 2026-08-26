"""`hvym-img <tool> --in ... --out ...` — run any registered tool locally.

Generic by construction: flags are derived from each tool's `InputModel`, so a
new tool gets a CLI for free (AGENTS.md §7).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydantic import BaseModel

from .core import registry
from .core.cache import ResultCache, hash_parts
from .core.config import Config
from .core.models import ModelCache
from .core.tool import Context, FileBytes, MediaResponse


def _help(text: str | None, fallback: str = "") -> str:
    """argparse %-expands help strings, so a description containing e.g. "74% of
    wall-clock" raises at --help time. Escape rather than constrain tool authors."""
    return (text or fallback).replace("%", "%%")


def _add_tool_args(parser: argparse.ArgumentParser, model: type[BaseModel]) -> tuple[str, ...]:
    file_fields: list[str] = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, FileBytes):
            file_fields.append(name)
            flag = "--in" if name == "image" else f"--{name.replace('_', '-')}"
            parser.add_argument(flag, dest=name, required=True, type=Path,
                                help=_help(field.description, f"{name} (file)"))
        elif annotation is bool:
            parser.add_argument(f"--{name.replace('_', '-')}", dest=name,
                                action="store_true", help=_help(field.description))
        else:
            parser.add_argument(
                f"--{name.replace('_', '-')}", dest=name,
                type=annotation if annotation in (int, float, str) else str,
                default=field.default if not field.is_required() else None,
                required=field.is_required(),
                help=_help(field.description),
            )
    return tuple(file_fields)


def _force_utf8_stdio() -> None:
    """Windows consoles default to cp1252, which cannot encode "≈" or "³" — both
    of which appear in tool descriptions. Reconfigure rather than restrict them."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # not a TextIOWrapper (e.g. captured)
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    registry.discover()

    parser = argparse.ArgumentParser(prog="hvym-img", description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="ignore any cached result")
    subparsers = parser.add_subparsers(dest="tool", metavar="TOOL")

    file_fields: dict[str, tuple[str, ...]] = {}
    for tool_cls in registry.all_tools():
        sub = subparsers.add_parser(tool_cls.name, help=_help(tool_cls.summary))
        file_fields[tool_cls.name] = _add_tool_args(sub, tool_cls.InputModel)
        sub.add_argument("--out", type=Path, required=True, help="output path")

    args = parser.parse_args(argv)
    if not args.tool:
        parser.print_help()
        return 2

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = Config.from_env()
    config.ensure_dirs()
    cache = ResultCache(config.cache_dir)
    ctx = Context(
        models=ModelCache(device=config.resolve_device()),
        cache=cache,
        workspace=config.workspace_dir,
        config=config,
    )

    tool = registry.get(args.tool)()
    payload = {}
    for name in tool.InputModel.model_fields:
        value = getattr(args, name, None)
        if value is None:
            continue
        payload[name] = value.read_bytes() if name in file_fields[args.tool] else value

    req = tool.InputModel(**payload)
    out_path: Path = args.out

    key = hash_parts(tool.cache_key_parts(req))
    hit = None if args.no_cache else cache.get(key)
    if hit is not None:
        out_path.write_bytes(hit.read())
        print(f"cache HIT  {key[:12]}  ->  {out_path}")
        return 0

    result = tool.run(req, ctx)
    if isinstance(result, MediaResponse):
        data = result.data
        cache.put(key, data, result.media_type)
    else:
        data = result.model_dump_json(indent=2).encode()
        cache.put(key, data, "application/json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    print(f"wrote {out_path} ({len(data)} bytes)  key={key[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
