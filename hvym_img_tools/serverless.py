"""RunPod Serverless entrypoint.

RunPod Serverless workers do **not** serve HTTP — they pull jobs from RunPod's
queue and call a handler. This adapts the same `Tool` contract the FastAPI server
uses, so a tool works in both deployments without knowing which it is in.

Job shape (`POST /v2/{endpoint}/runsync`):

    {"input": {"tool": "reangle", "image": "<base64>", "mc_resolution": 256}}

Response:

    {"output": {"data": "<base64 glb>", "media_type": "model/gltf-binary",
                "filename": "char.glb", "cached": false, "elapsed": 1.72}}

Binary comes back base64-encoded: a 729 KB `.glb` is ~970 KB encoded, comfortably
inside RunPod's response limit.

**Auth note.** Access here is gated by RunPod's own endpoint credentials, not by
`HVYM_API_KEY`. A RunPod account key grants *full account access*, so it must
never ship inside a desktop client — put `hvym_img_tools.proxy` in front and keep
the RunPod key server-side (docs/AUTH.md, docs/DEPLOY.md).
"""
from __future__ import annotations

import base64
import binascii
import logging
import time
from typing import Any

from .core import registry
from .core.cache import ResultCache, hash_parts
from .core.config import Config
from .core.models import ModelCache
from .core.server import configure_logging
from .core.tool import Context, MediaResponse

log = logging.getLogger(__name__)

#: Wire constant shared with the proxy's warm lease (hvym_img_tools/warm.py).
#: Duplicated rather than imported so the worker stays free of the proxy's HTTP
#: dependencies; tests assert the two definitions agree.
WARM_TOOL = "__warm__"

_CTX: Context | None = None
_TOOLS: dict[str, Any] = {}


def init() -> Context:
    """Build the context and warm every declared model.

    Called at import time in the container so warming happens while RunPod is
    still starting the worker, not inside the first job (~15 s of model load).
    """
    global _CTX
    if _CTX is not None:
        return _CTX

    configure_logging()
    config = Config.from_env()
    config.ensure_dirs()
    registry.discover()

    models = ModelCache(device=config.resolve_device())
    cache = ResultCache(config.cache_dir)
    ctx = Context(models=models, cache=cache, workspace=config.workspace_dir, config=config)

    for tool_cls in registry.all_tools():
        tool = tool_cls()
        _TOOLS[tool.name] = tool
        for key, loader in tool.model_loaders().items():
            if key not in models.registered():
                models.register(key, loader)

    if config.warm_on_startup:
        wanted = sorted({k for t in _TOOLS.values() for k in t.models_needed()})
        if wanted:
            try:
                times = models.warm(wanted)
                log.info("warmed models: %s", {k: round(v, 2) for k, v in times.items()})
            except Exception:  # noqa: BLE001 - degrade to lazy, don't kill the worker
                log.exception("model warm-up failed; falling back to lazy load")

        # Loaded != ready: kernels initialise on the first real forward pass.
        for tool in _TOOLS.values():
            try:
                started = time.perf_counter()
                tool.warmup(ctx)
                elapsed = time.perf_counter() - started
                if elapsed > 0.1:
                    log.info("warmed kernels for %s in %.2fs", tool.name, elapsed)
            except Exception:  # noqa: BLE001 - a failed warm-up must not kill the worker
                log.exception("kernel warm-up failed for %s; first request pays it", tool.name)

    _CTX = ctx
    log.info("serverless ready: tools=%s device=%s", sorted(_TOOLS), config.resolve_device())
    return ctx


def handler(job: dict) -> dict:
    """RunPod job handler. Returns `{"error": ...}` on failure, never raises."""
    started = time.perf_counter()
    ctx = init()

    payload = (job or {}).get("input") or {}
    if not isinstance(payload, dict):
        return {"error": "input must be an object"}

    name = payload.get("tool", "reangle")

    # A keepalive from a client warm lease (hvym_img_tools/warm.py, docs/WARMING.md).
    # Its entire purpose is to *be a completed job*, because that is what resets the
    # endpoint's idleTimeout and keeps this process alive with models resident.
    # Short-circuited before any pipeline work: at one ping every 8s, running
    # reconstruction here would burn GPU seconds for the whole lease and produce
    # nothing anyone reads.
    #
    # The literal is duplicated in warm.py rather than imported, so the worker does
    # not take a dependency on the proxy's HTTP stack. It is a wire constant; a test
    # asserts the two stay equal.
    if name == WARM_TOOL:
        return {
            "warm": True,
            "models": sorted(ctx.models.registered()),
            "device": ctx.config.resolve_device(),
            "elapsed": round(time.perf_counter() - started, 3),
        }

    tool = _TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}; available: {sorted(_TOOLS)}"}

    # Decode declared binary fields; everything else passes through as-is.
    data: dict[str, Any] = {}
    file_fields = type(tool).file_fields()
    for key, value in payload.items():
        if key == "tool":
            continue
        if key in file_fields:
            if not isinstance(value, str):
                return {"error": f"field {key!r} must be a base64 string"}
            try:
                data[key] = base64.b64decode(value, validate=True)
            except (binascii.Error, ValueError) as exc:
                return {"error": f"field {key!r} is not valid base64: {exc}"}
        else:
            data[key] = value

    missing = [f for f in file_fields if f not in data]
    if missing:
        return {"error": f"missing required field(s): {missing}"}

    try:
        req = type(tool).InputModel(**data)
    except Exception as exc:  # noqa: BLE001 - pydantic validation
        return {"error": f"invalid input: {exc}"}

    key = hash_parts(tool.cache_key_parts(req))
    hit = ctx.cache.get(key)
    if hit is not None:
        log.info("tool=%s cache=HIT key=%s", name, key[:12])
        return {
            "data": base64.b64encode(hit.read()).decode(),
            "media_type": hit.media_type,
            "cached": True,
            "elapsed": round(time.perf_counter() - started, 3),
            "cache_key": key,
        }

    try:
        result = tool.run(req, ctx)
    except Exception as exc:  # noqa: BLE001 - a bad job must not kill the worker
        log.exception("tool=%s failed", name)
        return {"error": f"{type(exc).__name__}: {exc}"}

    if isinstance(result, MediaResponse):
        ctx.cache.put(key, result.data, result.media_type)
        body = {
            "data": base64.b64encode(result.data).decode(),
            "media_type": result.media_type,
            "filename": result.filename,
        }
    else:
        payload_bytes = result.model_dump_json().encode()
        ctx.cache.put(key, payload_bytes, "application/json")
        body = {"json": result.model_dump(), "media_type": "application/json"}

    body |= {
        "cached": False,
        "elapsed": round(time.perf_counter() - started, 3),
        "cache_key": key,
        "tool_version": tool.version,
    }
    log.info("tool=%s cache=MISS elapsed=%.3fs", name, body["elapsed"])
    return body


def main() -> None:  # pragma: no cover - container entrypoint
    """Start the RunPod worker. `runpod` is imported here so the module stays
    importable (and testable) without the SDK installed."""
    import runpod

    init()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":  # pragma: no cover
    main()
