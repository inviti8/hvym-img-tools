"""Build the FastAPI app from the registry.

Every registered tool is mounted at `POST /tools/{name}` with its `InputModel`
fed to FastAPI, so OpenAPI is generated automatically (AGENTS.md §4, §8).

This is the only module that imports FastAPI — install the `server` extra.
"""
from __future__ import annotations

import inspect
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import (
    Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ValidationError

from . import registry
from .auth import API_KEY_HEADER, ApiKeyAuth, extract_key
from .cache import ResultCache, hash_parts
from .config import Config
from .models import ModelCache
from .tool import Context, MediaResponse, Tool

log = logging.getLogger(__name__)


def _make_auth_dependency(auth: ApiKeyAuth):
    """Gate a route on the API key.

    Always returns 401 (never 403) and never says *why* a key was rejected —
    "unknown key" vs "malformed" is free information for someone probing.
    """

    async def require_api_key(
        x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
        authorization: str | None = Header(default=None),
    ) -> None:
        if not auth.enabled:
            return
        if not auth.verify(extract_key(x_api_key, authorization)):
            raise HTTPException(
                status_code=401,
                detail="invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_api_key


def _build_endpoint(tool: Tool, ctx: Context):
    """Create a request handler whose signature FastAPI can introspect.

    Tools with binary fields get a multipart endpoint (files + form fields);
    tools without get a plain JSON body. Either way the tool only ever sees its
    own validated `InputModel`.
    """
    model = type(tool).InputModel
    file_fields = type(tool).file_fields()

    def _finish(result: Any, cache_key: str | None) -> Response:
        if isinstance(result, MediaResponse):
            if cache_key:
                ctx.cache.put(cache_key, result.data, result.media_type)
            headers = {"X-Cache": "MISS", "X-Tool-Version": tool.version}
            if cache_key:
                # Content address -- a natural asset id for a client library.
                headers["X-Cache-Key"] = cache_key
            if result.filename:
                headers["Content-Disposition"] = f'attachment; filename="{result.filename}"'
            return Response(content=result.data, media_type=result.media_type, headers=headers)
        if isinstance(result, BaseModel):
            return Response(
                content=result.model_dump_json(),
                media_type="application/json",
                headers={"X-Cache": "MISS", "X-Tool-Version": tool.version},
            )
        raise TypeError(
            f"{type(tool).__name__}.run returned {type(result).__name__}; "
            "expected a pydantic model or MediaResponse"
        )

    async def _handle(data: dict[str, Any]) -> Response:
        try:
            req = model(**data)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

        cache_key = hash_parts(tool.cache_key_parts(req))
        hit = ctx.cache.get(cache_key)
        if hit is not None:
            log.info("tool=%s cache=HIT key=%s", tool.name, cache_key[:12])
            return Response(
                content=hit.read(),
                media_type=hit.media_type,
                headers={
                    "X-Cache": "HIT",
                    "X-Tool-Version": tool.version,
                    "X-Cache-Key": cache_key,
                },
            )

        started = time.perf_counter()
        try:
            result = await run_in_threadpool(tool.run, req, ctx)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - surface as 500, keep server alive
            log.exception("tool=%s failed", tool.name)
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        elapsed = time.perf_counter() - started
        log.info("tool=%s cache=MISS elapsed=%.3fs key=%s", tool.name, elapsed, cache_key[:12])
        return _finish(result, cache_key)

    if file_fields:
        params = []
        for name, field in model.model_fields.items():
            if name in file_fields:
                params.append(inspect.Parameter(
                    name, inspect.Parameter.KEYWORD_ONLY,
                    annotation=UploadFile, default=File(...),
                ))
            else:
                default = Form(... if field.is_required() else field.default)
                params.append(inspect.Parameter(
                    name, inspect.Parameter.KEYWORD_ONLY,
                    annotation=field.annotation, default=default,
                ))

        async def endpoint(**kwargs: Any) -> Response:  # type: ignore[misc]
            data: dict[str, Any] = {}
            limit = ctx.config.max_upload_mb * 1024 * 1024
            for key, value in kwargs.items():
                if key in file_fields:
                    blob = await value.read()
                    if len(blob) > limit:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{key} is {len(blob)} bytes, limit is {limit}",
                        )
                    data[key] = blob
                else:
                    data[key] = value
            return await _handle(data)

        endpoint.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    else:
        async def endpoint(payload: model) -> Response:  # type: ignore[valid-type]
            return await _handle(payload.model_dump())

    endpoint.__name__ = f"run_{tool.name}"
    endpoint.__doc__ = tool.summary
    return endpoint


def configure_logging() -> None:
    """Make application logs visible regardless of entrypoint.

    `uvicorn hvym_img_tools.core.server:create_app --factory` never calls our
    `main()`, so relying on `basicConfig` there silently loses every application
    log: request timings, cache hit/miss, the front-view IoU, and the
    "auth is DISABLED" warning. Attach to our own namespace rather than the root
    logger so we neither fight uvicorn's config nor stomp on an embedding app.
    """
    import os

    app_log = logging.getLogger("hvym_img_tools")
    app_log.setLevel(os.environ.get("HVYM_LOG_LEVEL", "INFO").upper())
    if not app_log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        app_log.addHandler(handler)
    app_log.propagate = False


def create_app(config: Config | None = None, *, discover: bool = True) -> FastAPI:
    configure_logging()
    config = config or Config.from_env()
    config.ensure_dirs()

    if discover:
        registry.discover()

    auth = ApiKeyAuth.from_keys(config.api_keys)
    if not auth.enabled:
        log.warning(
            "API auth is DISABLED (HVYM_API_KEY unset) -- every caller can spend GPU "
            "time on this server. Set HVYM_API_KEY before exposing it publicly."
        )
    else:
        log.info("API auth enabled (%d key(s) accepted)", len(auth.keys))
    guard = [Depends(_make_auth_dependency(auth))]

    device = config.resolve_device()
    models = ModelCache(device=device)
    cache = ResultCache(config.cache_dir)
    ctx = Context(models=models, cache=cache, workspace=config.workspace_dir, config=config)

    instances: list[Tool] = []

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Warm every declared model once, at startup, so no request pays the
        # load cost (AGENTS.md §4). Measured cold load: ~13.7s (BENCHMARK.md §1).
        if config.warm_on_startup:
            wanted = sorted({k for t in instances for k in t.models_needed()})
            if wanted:
                try:
                    times = models.warm(wanted)
                    log.info("warmed models: %s", {k: round(v, 2) for k, v in times.items()})
                except Exception:  # noqa: BLE001 - degrade to lazy load, don't fail startup
                    log.exception("model warm-up failed; falling back to lazy load")
        yield

    app = FastAPI(
        title="hvym-img-tools",
        summary="Self-hosted, AI-augmented image tools",
        version=__import__("hvym_img_tools").__version__,
        lifespan=lifespan,
    )
    app.state.context = ctx
    app.state.config = config

    for tool_cls in registry.all_tools():
        tool = tool_cls()
        instances.append(tool)
        # Register loaders now, at build time, so startup warm-up can find them.
        for key, loader in tool.model_loaders().items():
            if key not in models.registered():
                models.register(key, loader)
        app.add_api_route(
            f"/tools/{tool.name}",
            _build_endpoint(tool, ctx),
            methods=["POST"],
            summary=tool.summary,
            tags=[tool.name],
            response_class=Response,
            dependencies=guard,
        )
        log.info("mounted POST /tools/%s (v%s)", tool.name, tool.version)

    @app.get("/tools", tags=["meta"], dependencies=guard)
    def list_tools() -> dict:
        return {"device": device, "tools": registry.describe_all()}

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict:
        # Intentionally unauthenticated so orchestrator health probes work.
        # Exposes no secrets and costs no GPU time.
        return {
            "status": "ok",
            "auth": auth.enabled,
            "device": device,
            "tools": registry.names(),
            "models_loaded": models.loaded(),
            "cache": cache.stats(),
        }

    # --- warm lease: a truthful no-op here ---------------------------------
    # A persistent box does not scale to zero, so it is always warm and there is
    # nothing to lease. Answering honestly (rather than 404ing) is what lets
    # Inkternity ship ONE code path: the client holds a lease unconditionally,
    # and against this deployment every call simply says "already warm".
    # docs/WARMING.md, docs/CLIENT.md.
    from ..warm import always_warm_view  # noqa: PLC0415 - avoids a cycle at import

    @app.post("/warm", tags=["warm"], dependencies=guard)
    async def warm_acquire(request: Request) -> dict:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - an empty body is valid
            body = {}
        lease_id = (body or {}).get("lease_id") if isinstance(body, dict) else None
        return always_warm_view(str(lease_id) if lease_id else None)

    @app.get("/warm", tags=["warm"])
    def warm_status() -> dict:
        return always_warm_view()

    @app.delete("/warm", tags=["warm"], dependencies=guard)
    def warm_release() -> dict:
        return always_warm_view()

    return app


def main() -> None:
    """Entry point for `uv run hvym-img-serve`."""
    import os

    import uvicorn

    configure_logging()
    uvicorn.run(
        create_app(),
        host=os.environ.get("HVYM_HOST", "0.0.0.0"),
        port=int(os.environ.get("HVYM_PORT", "8000")),
    )
