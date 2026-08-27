"""Authenticating proxy in front of a RunPod Serverless endpoint.

**Why this exists.** Calling RunPod Serverless directly requires a RunPod API key,
and an account key grants *full account access* — create pods, spend the balance,
delete things. Shipping one inside a desktop binary would be strictly worse than
the scoped key we issue ourselves. This proxy keeps the RunPod key server-side and
exposes only "ask for a mesh".

**It mirrors the direct server's HTTP contract exactly** (`POST /tools/{name}`,
multipart in, binary out, `X-Cache` / `X-Tool-Version` headers). Inkternity's
client code is therefore identical whether it talks to a persistent pod running
`core.server` or to this proxy in front of serverless — the deployment can change
without touching the client.

The proxy does no GPU work, so it can run on the cheapest always-on box available.

    HVYM_API_KEY=...            # the scoped key Inkternity holds
    RUNPOD_API_KEY=...          # NEVER leaves this process
    RUNPOD_ENDPOINT_ID=...      # the serverless endpoint
    uv run hvym-img-proxy
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response

from .core.auth import API_KEY_HEADER, ApiKeyAuth, extract_key
from .core.config import Config
from .core.server import configure_logging
from .warm import WarmPool

log = logging.getLogger(__name__)

#: RunPod's synchronous endpoint. Cold start can take *minutes* -- the worker has
#: to pull a ~6.5GB image before it loads a single model -- so the budget below is
#: whole-operation wall clock, not per-HTTP-call. Warm requests return in ~2s.
RUNPOD_BASE = "https://api.runpod.ai/v2"
DEFAULT_TIMEOUT = float(os.environ.get("HVYM_PROXY_TIMEOUT", "600"))

#: /runsync does not block indefinitely: RunPod caps it server-side (~90s) and
#: then hands back the job still IN_QUEUE rather than the result. That is not an
#: error and not an edge case -- a scale-from-zero cold start exceeds the cap
#: every time -- so a queued job is polled to completion via /status/{id}.
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"})
POLL_INITIAL = 1.0
POLL_MAX = 5.0


def _auth_dependency(auth: ApiKeyAuth):
    async def require_api_key(request: Request) -> None:
        if not auth.enabled:
            return
        presented = extract_key(
            request.headers.get(API_KEY_HEADER), request.headers.get("authorization")
        )
        if not auth.verify(presented):
            raise HTTPException(
                status_code=401,
                detail="invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_api_key


def create_app() -> FastAPI:
    configure_logging()
    config = Config.from_env()

    auth = ApiKeyAuth.from_keys(config.api_keys)
    if not auth.enabled:
        log.warning(
            "Proxy auth is DISABLED (HVYM_API_KEY unset) -- anyone who reaches this "
            "proxy can spend GPU time. Set HVYM_API_KEY before exposing it."
        )

    runpod_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    # Tools live on separate serverless endpoints (docs/tools/mesh.md §5), so a
    # single RUNPOD_ENDPOINT_ID is no longer enough. It stays as the default, and
    # RUNPOD_ENDPOINT_ID_<TOOL> overrides per tool -- so an existing deployment
    # keeps working unchanged and only gains routing when it sets the extras.
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
    tool_endpoints = {
        key[len("RUNPOD_ENDPOINT_ID_"):].lower(): value.strip()
        for key, value in os.environ.items()
        if key.startswith("RUNPOD_ENDPOINT_ID_") and value.strip()
    }

    def endpoint_for(tool: str) -> str:
        return tool_endpoints.get(tool.lower(), endpoint_id)

    if not runpod_key or not (endpoint_id or tool_endpoints):
        # Fail loudly at startup rather than 500 on the first real request.
        log.error("RUNPOD_API_KEY and at least one RUNPOD_ENDPOINT_ID must be set")
    if tool_endpoints:
        log.info("per-tool endpoints: %s", sorted(tool_endpoints))

    guard = [Depends(_auth_dependency(auth))]

    # Warm leases (docs/WARMING.md). Built before the app so the keepalive loop
    # can be torn down from a lifespan handler: if this process dies the pings
    # stop and the worker sleeps on its own, which is the whole reason a lease
    # beats a workersMin switch here.
    # A lease should warm the endpoint the artist is about to use, not every
    # endpoint we have (docs/WARMING.md): warm time is the metered unit, so
    # warming both would double the bill for no benefit. One pool per endpoint,
    # created on demand.
    pools: dict[str, WarmPool] = {}

    def pool_for(tool: str) -> WarmPool:
        target = endpoint_for(tool)
        if target not in pools:
            pools[target] = WarmPool(runpod_key, target)
        return pools[target]

    pool = pool_for("reangle")          # default endpoint's pool

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for p in list(pools.values()):
            await p.shutdown()

    app = FastAPI(title="hvym-img-tools proxy", version="0.1.0", lifespan=lifespan)
    app.state.warm_pool = pool

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict:
        # Unauthenticated for orchestrator probes; reports only whether things are
        # configured, never the values.
        return {
            "status": "ok",
            "mode": "proxy",
            "auth": auth.enabled,
            "runpod_configured": bool(runpod_key and (endpoint_id or tool_endpoints)),
            "endpoint_id": endpoint_id or None,
            "tool_endpoints": sorted(tool_endpoints) or None,
        }

    # ---------------------------------------------------------------- warming
    # Gated by the same scoped key as /tools/{name}: a HVYM_API_KEY holder can
    # ask for warmth and nothing else. The RunPod account key stays in this
    # process (docs/AUTH.md, docs/WARMING.md).
    async def _lease_id_from(request: Request) -> tuple[str | None, str]:
        """Body is optional -- a first POST /warm legitimately has none."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - empty or non-JSON body is fine here
            return None, "", "reangle"
        if not isinstance(body, dict):
            return None, "", "reangle"
        lease_id = body.get("lease_id")
        label = body.get("label") or body.get("client") or ""
        tool = body.get("tool") or "reangle"
        return (str(lease_id) if lease_id else None), str(label)[:64], str(tool)[:32]

    @app.post("/warm", dependencies=guard, tags=["warm"])
    async def warm_acquire(request: Request) -> dict:
        if not runpod_key or not endpoint_id:
            raise HTTPException(status_code=503, detail="proxy is not configured")
        lease_id, label, tool = await _lease_id_from(request)
        return await pool_for(tool).acquire(lease_id, label)

    @app.get("/warm", tags=["warm"])
    async def warm_status(tool: str = "reangle") -> dict:
        # Unauthenticated on purpose: it is a read-only indicator that spends
        # nothing, and the UI wants it before the artist has a lease. It reports
        # no key, no endpoint URL, and cannot start a worker.
        if not runpod_key or not endpoint_id:
            raise HTTPException(status_code=503, detail="proxy is not configured")
        return await pool_for(tool).status()

    @app.delete("/warm", dependencies=guard, tags=["warm"])
    async def warm_release(request: Request) -> dict:
        if not runpod_key or not endpoint_id:
            raise HTTPException(status_code=503, detail="proxy is not configured")
        lease_id, _, tool = await _lease_id_from(request)
        if not lease_id:
            raise HTTPException(status_code=422, detail="lease_id is required to release")
        return await pool_for(tool).release(lease_id)

    @app.post("/tools/{name}", dependencies=guard, response_class=Response)
    async def call_tool(name: str, request: Request) -> Response:
        target = endpoint_for(name)
        if not runpod_key or not target:
            raise HTTPException(status_code=503, detail="proxy is not configured")

        form = await request.form()
        payload: dict[str, Any] = {"tool": name}
        limit = config.max_upload_mb * 1024 * 1024
        for key, value in form.items():
            if hasattr(value, "read"):  # an uploaded file
                blob = await value.read()
                if len(blob) > limit:
                    raise HTTPException(
                        status_code=413, detail=f"{key} is {len(blob)} bytes, limit is {limit}"
                    )
                payload[key] = base64.b64encode(blob).decode()
            else:
                payload[key] = value

        started = time.perf_counter()
        auth_header = {"Authorization": f"Bearer {runpod_key}"}

        # A real request is itself a job, so it resets the worker's idleTimeout.
        # Telling the warm pool suppresses its keepalive for the duration: firing
        # one alongside this request is pure contention, and was measured letting
        # RunPod dispatch the request to a second, cold worker.
        warm = pool_for(name)
        warm.request_started()

        def _check(resp: httpx.Response) -> dict:
            if resp.status_code >= 400:
                log.error("runpod returned %s", resp.status_code)
                raise HTTPException(status_code=502, detail=f"upstream status {resp.status_code}")
            return resp.json()

        try:
            # One client for the whole job so polling reuses the connection.
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                body = _check(await client.post(
                    f"{RUNPOD_BASE}/{target}/runsync",
                    headers=auth_header,
                    json={"input": payload},
                ))

                status = body.get("status")
                job_id = body.get("id")
                delay = POLL_INITIAL
                while status not in TERMINAL_STATUSES and status is not None and job_id:
                    if time.perf_counter() - started > DEFAULT_TIMEOUT:
                        raise HTTPException(
                            status_code=504,
                            detail=f"job {status} after {DEFAULT_TIMEOUT:.0f}s",
                        )
                    if delay == POLL_INITIAL:
                        log.info("tool=%s job queued upstream, polling for result", name)
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.5, POLL_MAX)
                    body = _check(await client.get(
                        f"{RUNPOD_BASE}/{target}/status/{job_id}", headers=auth_header
                    ))
                    status = body.get("status")
        except HTTPException:
            raise
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail="upstream timed out") from exc
        except httpx.HTTPError as exc:
            # Deliberately does not echo the upstream URL or key material.
            raise HTTPException(status_code=502, detail=f"upstream error: {type(exc).__name__}") from exc
        finally:
            warm.request_finished()

        if status not in (None, "COMPLETED"):
            detail = body.get("error") or f"job status {status}"
            raise HTTPException(status_code=502, detail=str(detail)[:500])

        output = body.get("output") or {}
        if "error" in output:
            raise HTTPException(status_code=500, detail=str(output["error"])[:500])

        elapsed = time.perf_counter() - started
        headers = {
            "X-Cache": "HIT" if output.get("cached") else "MISS",
            "X-Upstream-Elapsed": str(output.get("elapsed", "")),
            "X-Proxy-Elapsed": f"{elapsed:.3f}",
        }
        if output.get("tool_version"):
            headers["X-Tool-Version"] = str(output["tool_version"])
        if output.get("cache_key"):
            # sha256(image + params) -- already a content address, so it is a
            # natural asset id for a client-side library (docs/tools/mesh.md §6).
            headers["X-Cache-Key"] = str(output["cache_key"])

        if "json" in output:
            import json as _json

            return Response(
                content=_json.dumps(output["json"]),
                media_type="application/json",
                headers=headers,
            )

        encoded = output.get("data")
        if not encoded:
            raise HTTPException(status_code=502, detail="upstream returned no data")
        try:
            blob = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=502, detail="upstream returned invalid base64") from exc

        if output.get("filename"):
            headers["Content-Disposition"] = f'attachment; filename="{output["filename"]}"'
        log.info("tool=%s proxied in %.3fs (upstream %s)", name, elapsed, output.get("elapsed"))
        return Response(
            content=blob,
            media_type=output.get("media_type", "application/octet-stream"),
            headers=headers,
        )

    return app


def main() -> None:  # pragma: no cover - entrypoint
    import uvicorn

    configure_logging()
    uvicorn.run(
        create_app(),
        host=os.environ.get("HVYM_HOST", "0.0.0.0"),
        port=int(os.environ.get("HVYM_PORT", "8080")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
