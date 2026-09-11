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

**Billing** (docs/X402_BILLING.md) rides on top, off by default. When enabled,
identity stops being the shared `X-API-Key` and becomes a per-request ed25519
signature from the artist's own wallet (`core.identity`), and a live paid window
becomes a precondition for spending GPU time (`billing`). Both are behind flags
so this ships, and is observed, before it enforces.

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
from fastapi.responses import JSONResponse

from .billing import Billing, PaymentInvalid, PaymentRequired
from .core.auth import API_KEY_HEADER, ApiKeyAuth, extract_key
from .core.config import Config
from .core.identity import (
    AUTH_HEADER,
    PUBKEY_HEADER,
    IdentityError,
    IdentityVerifier,
    SignedIdentity,
)
from .core.server import configure_logging
from .warm import WarmPool

log = logging.getLogger(__name__)

#: RunPod's synchronous endpoint. Cold start can take *minutes* -- the worker has
#: to pull a ~6.5GB image before it loads a single model -- so the budget below is
#: whole-operation wall clock, not per-HTTP-call. Warm requests return in ~2s.
RUNPOD_BASE = "https://api.runpod.ai/v2"

#: Was 600s, which a measured 547s cold mesh start cleared by 53 seconds. It also
#: quietly outranked the nginx fix: with proxy_read_timeout raised to 900s, this
#: was the tighter of the two and gave up first.
#:
#: The ordering is deliberate, not incidental. This budget must stay BELOW nginx's
#: read timeout so a genuinely stuck job returns our JSON {"detail": ...} and the
#: artist sees a real message -- if nginx wins the race they get its HTML 504
#: instead, which is what a mesh cold start produced before any of this was fixed.
#: Keep the gap: proxy 840s < nginx 900s.
DEFAULT_TIMEOUT = float(os.environ.get("HVYM_PROXY_TIMEOUT", "840"))

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

    # ---------------------------------------------------------------- billing
    # docs/X402_BILLING.md. Inert until its flags are set: with both off this
    # only *observes* -- it verifies whatever signatures arrive, logs what it
    # would have refused, and serves the request exactly as before. That is the
    # whole point of the phased rollout (§2): the meter can be watched running
    # against real traffic before it is allowed to turn anyone away.
    billing = Billing()
    verifier = IdentityVerifier(window_s=billing.config.sign_window_s)
    if billing.require_payment:
        broken = billing.config.misconfiguration()
        if broken:
            # Refusing to start beats 402-ing every artist with a challenge that
            # names no payee: nobody could pay it, and the failure would look
            # like a client bug.
            raise RuntimeError(f"HVYM_REQUIRE_PAYMENT is on but {broken}")
        log.info(
            "billing ENFORCED: %s %s per %.0fs window to %s on %s (db=%s)",
            billing.config.amount, billing.config.asset, billing.config.window_s,
            billing.config.payee[:8], billing.config.network, billing.config.db_path,
        )
    elif billing.require_identity:
        log.info("signed identity ENFORCED; payment not enforced (observing only)")
    else:
        log.info("billing observing only; set HVYM_REQUIRE_SIGNED_IDENTITY to enforce identity")

    async def identity_for(
        request: Request, *, tool: str, lease_id: str = ""
    ) -> SignedIdentity | None:
        """Verify the signed identity on a request, honouring the rollout flag.

        Returns None when there is no usable identity *and* one is not required
        yet -- callers then fall back to today's `X-API-Key`-only behaviour. The
        path verified is the one this app sees (`request.url.path`), which is
        also what the client signs; if an nginx prefix is ever added in front,
        both sides have to learn about it together.
        """
        pubkey_header = request.headers.get(PUBKEY_HEADER)
        auth_header = request.headers.get(AUTH_HEADER)
        try:
            return verifier.verify(
                pubkey_header=pubkey_header,
                auth_header=auth_header,
                method=request.method,
                path=request.url.path,
                tool=tool,
                lease_id=lease_id,
            )
        except IdentityError as exc:
            if billing.require_identity:
                raise HTTPException(status_code=401, detail=f"signed identity: {exc}") from exc
            if IdentityVerifier.presented(pubkey_header, auth_header):
                # A client that meant to sign and got it wrong is a bug worth
                # seeing now, while it is still harmless.
                log.warning("signed identity rejected (not enforced): %s", exc)
            else:
                log.debug("unsigned request to %s (identity not enforced)", request.url.path)
            return None

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
            pools[target] = WarmPool(runpod_key, target, on_warm=billing.on_warm)
        return pools[target]

    pool = pool_for("reangle")          # default endpoint's pool

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for p in list(pools.values()):
            await p.shutdown()
        billing.close()

    app = FastAPI(title="hvym-img-tools proxy", version="0.4.0", lifespan=lifespan)
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
            # Which half of the rollout is live. Deliberately booleans only: no
            # payee, no price, no db path -- healthz stays free of anything an
            # operator would not want on an open endpoint.
            "signed_identity": billing.require_identity,
            "payment": billing.require_payment,
        }

    @app.exception_handler(PaymentRequired)
    async def _payment_required(_request: Request, exc: PaymentRequired) -> JSONResponse:
        # The x402 challenge is the response *body*, not a `detail` string: the
        # client parses `x402` to build a payment (docs/X402_BILLING.md §4.1), so
        # burying it inside FastAPI's default envelope would break the contract.
        body: dict[str, Any] = {"detail": exc.detail}
        if exc.challenge:
            body["x402"] = exc.challenge
        return JSONResponse(status_code=402, content=body)

    # ---------------------------------------------------------------- warming
    # Gated by the same scoped key as /tools/{name}: a HVYM_API_KEY holder can
    # ask for warmth and nothing else. The RunPod account key stays in this
    # process (docs/AUTH.md, docs/WARMING.md).
    async def _lease_id_from(request: Request) -> tuple[str | None, str, str]:
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
        identity = await identity_for(request, tool=tool, lease_id=lease_id or "")
        if identity is not None:
            # The verified pubkey *is* the metering label. A client-supplied
            # `label` is never trusted for it: attribution the payer can choose
            # is attribution someone else can wear.
            label = identity.pubkey
            billing.require(identity.pubkey)
        elif billing.require_payment:
            billing.require(None)
        return await pool_for(tool).acquire(lease_id, label)

    @app.get("/warm", tags=["warm"])
    async def warm_status(tool: str = "reangle") -> dict:
        # Unauthenticated on purpose: it is a read-only indicator that spends
        # nothing, and the UI wants it before the artist has a lease. It reports
        # no key, no endpoint URL, and cannot start a worker. Payment does not
        # gate it either (docs/X402_BILLING.md §3): an unpaid client still has to
        # be able to read "cold" to know what it would be buying.
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
        identity = await identity_for(request, tool=tool, lease_id=lease_id)
        if identity is not None and billing.require_identity:
            # Releasing never costs anything, so it needs no paid window -- but
            # it does need ownership. Dropping another artist's lease would put
            # out a worker they are still paying to keep warm.
            owner = pool_for(tool).owner_of(lease_id)
            if owner and owner != identity.pubkey:
                raise HTTPException(
                    status_code=403, detail="that lease belongs to another identity"
                )
        return await pool_for(tool).release(lease_id)

    # ------------------------------------------------------------------ x402
    @app.get("/warm/price", tags=["warm"])
    def warm_price() -> dict:
        """Open and read-only, so the UI can show a price without taking a 402.

        It issues no quote and touches no state: asking what something costs must
        not itself be a billable event, or the price indicator becomes a meter.
        """
        return billing.price_view()

    @app.post("/warm/pay", dependencies=guard, tags=["warm"])
    async def warm_pay(request: Request) -> dict:
        """Settle a Stellar payment the client already submitted itself.

        We never move funds: the artist self-custodies and broadcasts, and this
        route only confirms on Horizon that the payment landed, then extends a
        window. A compromised proxy therefore has no path to anyone's money.
        """
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - the proof may ride entirely in headers
            body = {}
        if not isinstance(body, dict):
            body = {}
        tx_hash = (request.headers.get("X-Payment") or body.get("tx") or "").strip()
        price_id = str(body.get("price_id") or "").strip()
        tool = str(body.get("tool") or "reangle")[:32]

        identity = await identity_for(
            request, tool=tool, lease_id=str(body.get("lease_id") or "")
        )
        if identity is None:
            # Unlike the gated routes, this one has no un-identified mode: a
            # payment with no identity is money credited to nobody.
            raise HTTPException(
                status_code=401,
                detail=f"{PUBKEY_HEADER}/{AUTH_HEADER} are required to settle a payment",
            )
        if not tx_hash:
            raise HTTPException(
                status_code=422, detail="X-Payment (a Stellar tx hash) is required"
            )
        broken = billing.config.misconfiguration()
        if broken:
            raise HTTPException(status_code=503, detail=f"billing is not configured: {broken}")

        try:
            settled = await billing.settle_payment(identity.pubkey, tx_hash, price_id)
        except PaymentInvalid as exc:
            # A fresh challenge rides along so a client that paid the wrong thing
            # can correct it in one round trip rather than asking again.
            raise PaymentRequired(billing.challenge(identity.pubkey), str(exc)) from exc

        # Hand back the warm view too, so the client sees state/expires_at
        # immediately instead of having to poll for it (§4.2).
        settled["warm"] = await pool_for(tool).status()
        return settled

    @app.post("/tools/{name}", dependencies=guard, response_class=Response)
    async def call_tool(name: str, request: Request) -> Response:
        target = endpoint_for(name)
        if not runpod_key or not target:
            raise HTTPException(status_code=503, detail="proxy is not configured")

        # Charged before the upload is read, not after: a 402 should cost the
        # artist one round trip, not an image body they have to send twice. The
        # signature covers the path/tool/time/nonce -- the multipart body is
        # deliberately unsigned (docs/X402_BILLING.md §2).
        identity = await identity_for(request, tool=name)
        if identity is not None:
            billing.require(identity.pubkey)
        elif billing.require_payment:
            billing.require(None)

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
