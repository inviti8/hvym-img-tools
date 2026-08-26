"""The `Tool` contract — the entire framework surface a tool touches.

A tool declares its name, typed I/O, which models it needs warmed, and a `run()`.
It knows nothing about other tools, the server, or how it is mounted.

Keep this module CPU-importable: no torch, no GPU imports at module load
(AGENTS.md §6), so the registry and tests run without a GPU.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:  # avoid import cycles; these are only needed for typing
    from .cache import ResultCache
    from .config import Config
    from .models import ModelCache


class FileBytes(bytes):
    """Marks an `InputModel` field as a binary upload.

    The registry turns these into multipart file parameters and hands `run()`
    the raw bytes, so a tool never deals with HTTP.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        # Validate as plain bytes, then re-wrap so `file_fields()` can still
        # identify the field by its annotation.
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(cls, core_schema.bytes_schema())

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: Any, handler: Any) -> Any:
        json_schema = handler(schema)
        json_schema.update(type="string", format="binary")
        return json_schema


@dataclass(slots=True)
class MediaResponse:
    """A binary result (a `.glb`, a `.png`) returned with its content type."""

    data: bytes
    media_type: str
    filename: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, (bytes, bytearray)):
            raise TypeError(f"MediaResponse.data must be bytes, got {type(self.data).__name__}")


@dataclass(slots=True)
class Context:
    """Everything a tool is given at run time.

    Carries the model cache, the result cache, a temp workspace and config —
    so `run()` never reaches for globals.
    """

    models: "ModelCache"
    cache: "ResultCache"
    workspace: Path
    config: "Config"
    extras: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """One capability. Subclass, set the class attributes, implement `run()`."""

    name: ClassVar[str]
    summary: ClassVar[str]
    version: ClassVar[str] = "0.1.0"

    InputModel: ClassVar[type[BaseModel]]
    #: A pydantic model (JSON response) or `MediaResponse` (binary response).
    OutputModel: ClassVar[type[BaseModel] | type[MediaResponse]]

    def model_loaders(self) -> dict[str, Any]:
        """Map of `{model_key: loader}` this tool needs.

        Declared, not registered on first use: the server registers these when
        the app is built, *before* startup warm-up runs. Registering inside
        `run()` instead means warm-up finds nothing and every model loads on the
        first request — which is exactly the cold-start cost `ModelCache` exists
        to avoid.
        """
        return {}

    def models_needed(self) -> list[str]:
        """Keys the `ModelCache` should warm before serving this tool.

        Defaults to everything `model_loaders()` declares; override only to warm
        a subset (e.g. leave a rarely-used model lazy).
        """
        return sorted(self.model_loaders())

    def cache_key_parts(self, req: BaseModel) -> list[bytes]:
        """Bytes identifying this request for the result cache.

        Default covers every field, with `FileBytes` hashed by content. Override
        when some field must not affect identity.
        """
        parts: list[bytes] = [self.name.encode(), self.version.encode()]
        for key in sorted(type(req).model_fields):
            value = getattr(req, key)
            parts.append(key.encode())
            parts.append(value if isinstance(value, (bytes, bytearray)) else repr(value).encode())
        return parts

    @abstractmethod
    def run(self, req: Any, ctx: Context) -> BaseModel | MediaResponse:
        """Do the work. Called off the event loop, so blocking is fine."""

    # -- introspection -------------------------------------------------------

    @classmethod
    def returns_media(cls) -> bool:
        return isinstance(cls.OutputModel, type) and issubclass(cls.OutputModel, MediaResponse)

    @classmethod
    def file_fields(cls) -> tuple[str, ...]:
        return tuple(
            n for n, f in cls.InputModel.model_fields.items()
            if isinstance(f.annotation, type) and issubclass(f.annotation, FileBytes)
        )

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {
            "name": cls.name,
            "summary": cls.summary,
            "version": cls.version,
            "returns_media": cls.returns_media(),
            "file_fields": list(cls.file_fields()),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} v{self.version}>"
