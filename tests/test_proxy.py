"""Proxy tests — the RunPod upstream is mocked, so no network and no GPU.

The property that matters most: the proxy must present the *same* HTTP contract
as the direct server, so Inkternity's client is identical either way. And the
RunPod key must never appear in a response.
"""
from __future__ import annotations

import base64

import httpx
import pytest
from fastapi.testclient import TestClient

from hvym_img_tools import proxy as proxy_mod

GOOD = "proxy-test-key-long-enough-0001"
RUNPOD_KEY = "runpod-secret-must-never-leak-9999"
GLB = b"glTF" + b"\x00" * 64


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("HVYM_API_KEY", GOOD)
    monkeypatch.setenv("RUNPOD_API_KEY", RUNPOD_KEY)
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "ep123")
    monkeypatch.setenv("HVYM_DEVICE", "cpu")


class FakeUpstream:
    """Stands in for RunPod, capturing what the proxy actually sent."""

    def __init__(self, payload=None, status=200):
        self.payload = payload if payload is not None else {
            "status": "COMPLETED",
            "output": {
                "data": base64.b64encode(GLB).decode(),
                "media_type": "model/gltf-binary",
                "filename": "char.glb",
                "cached": False,
                "elapsed": 1.72,
                "tool_version": "0.1.0",
            },
        }
        self.status = status
        self.seen: dict = {}

    def install(self, monkeypatch, raises=None, queued_polls=0):
        """queued_polls>0 makes /runsync answer IN_QUEUE, as RunPod really does
        once a job outlives its ~90s server-side cap, and hand the result over
        only after that many /status polls."""
        upstream = self
        upstream.polls = 0

        class _Client:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers=None, json=None):
                if raises is not None:
                    raise raises
                upstream.seen = {"url": url, "headers": headers or {}, "json": json or {}}
                body = upstream.payload
                if queued_polls:
                    body = {"id": "job-abc", "status": "IN_QUEUE"}
                return httpx.Response(upstream.status, json=body,
                                      request=httpx.Request("POST", url))

            async def get(self, url, headers=None):
                upstream.polls += 1
                upstream.polled_url = url
                if upstream.polls < queued_polls:
                    body = {"id": "job-abc", "status": "IN_PROGRESS"}
                else:
                    body = dict(upstream.payload, id="job-abc")
                return httpx.Response(200, json=body,
                                      request=httpx.Request("GET", url))

        monkeypatch.setattr(proxy_mod.httpx, "AsyncClient", _Client)
        return upstream


def _client():
    return TestClient(proxy_mod.create_app())


def test_healthz_open_and_reports_config(env):
    body = _client().get("/healthz").json()
    assert body["status"] == "ok"
    assert body["mode"] == "proxy"
    assert body["auth"] is True
    assert body["runpod_configured"] is True


def test_healthz_never_exposes_the_runpod_key(env):
    text = _client().get("/healthz").text
    assert RUNPOD_KEY not in text


def test_requires_our_api_key(env, monkeypatch):
    FakeUpstream().install(monkeypatch)
    resp = _client().post("/tools/reangle", files={"image": ("a.png", b"x", "image/png")})
    assert resp.status_code == 401


def test_forwards_and_returns_binary(env, monkeypatch):
    up = FakeUpstream().install(monkeypatch)
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"RAW", "image/png")},
        data={"mc_resolution": "256"},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 200
    assert resp.content == GLB
    assert resp.headers["content-type"] == "model/gltf-binary"
    assert resp.headers["x-cache"] == "MISS"
    assert resp.headers["x-tool-version"] == "0.1.0"
    assert "char.glb" in resp.headers["content-disposition"]

    # what actually went upstream
    sent = up.seen["json"]["input"]
    assert sent["tool"] == "reangle"
    assert base64.b64decode(sent["image"]) == b"RAW"
    assert sent["mc_resolution"] == "256"
    assert up.seen["headers"]["Authorization"] == f"Bearer {RUNPOD_KEY}"
    assert up.seen["url"].endswith("/ep123/runsync")


def test_cache_hit_header_is_propagated(env, monkeypatch):
    FakeUpstream(payload={
        "status": "COMPLETED",
        "output": {"data": base64.b64encode(GLB).decode(), "cached": True,
                   "media_type": "model/gltf-binary"},
    }).install(monkeypatch)
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.headers["x-cache"] == "HIT"


def test_bearer_form_also_accepted(env, monkeypatch):
    FakeUpstream().install(monkeypatch)
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"Authorization": f"Bearer {GOOD}"},
    )
    assert resp.status_code == 200


def test_tool_error_becomes_500(env, monkeypatch):
    FakeUpstream(payload={"status": "COMPLETED",
                          "output": {"error": "backbone exploded"}}).install(monkeypatch)
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 500
    assert "backbone exploded" in resp.json()["detail"]


def test_failed_job_becomes_502(env, monkeypatch):
    FakeUpstream(payload={"status": "FAILED", "error": "worker died"}).install(monkeypatch)
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 502


def test_upstream_timeout_becomes_504(env, monkeypatch):
    FakeUpstream().install(monkeypatch, raises=httpx.TimeoutException("slow"))
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 504


def test_upstream_error_does_not_leak_key_or_url(env, monkeypatch):
    FakeUpstream().install(monkeypatch, raises=httpx.ConnectError("boom"))
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 502
    assert RUNPOD_KEY not in resp.text
    assert "api.runpod.ai" not in resp.text


def test_unconfigured_proxy_is_503(monkeypatch):
    monkeypatch.setenv("HVYM_API_KEY", GOOD)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
    monkeypatch.setenv("HVYM_DEVICE", "cpu")
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 503


# --- queued jobs -----------------------------------------------------------
# A scale-from-zero cold start ALWAYS outlives RunPod's ~90s /runsync cap: the
# worker pulls a ~6.5GB image before it loads a model. runsync then returns the
# job still IN_QUEUE, which the proxy used to report as a 502 -- so the very
# first request after every scale-to-zero failed. Live deploy caught this; the
# mocked upstream above never produced it.

@pytest.fixture(autouse=True)
def _no_poll_sleep(monkeypatch):
    """Poll backoff is real seconds; tests should not pay it.

    Collapses the backoff constants rather than monkeypatching `asyncio.sleep`.
    That call mutated the *real* asyncio module for every test in this file --
    including the warm-lease tests below, whose keepalive loop would then spin
    hot instead of sleeping between pings.
    """
    monkeypatch.setattr(proxy_mod, "POLL_INITIAL", 0.0)
    monkeypatch.setattr(proxy_mod, "POLL_MAX", 0.0)


def test_queued_job_is_polled_to_completion(env, monkeypatch):
    up = FakeUpstream().install(monkeypatch, queued_polls=3)
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 200, resp.text
    assert resp.content == GLB
    assert up.polls == 3
    assert up.polled_url.endswith("/ep123/status/job-abc")


def test_polling_gives_up_and_reports_504(env, monkeypatch):
    monkeypatch.setattr(proxy_mod, "DEFAULT_TIMEOUT", -1.0)  # budget already spent
    FakeUpstream().install(monkeypatch, queued_polls=99)
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 504
    assert "IN_QUEUE" in resp.json()["detail"]


def test_job_that_fails_while_queued_becomes_502(env, monkeypatch):
    FakeUpstream(payload={"status": "FAILED", "error": "worker died"}).install(
        monkeypatch, queued_polls=1)
    resp = _client().post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 502
    assert "worker died" in resp.json()["detail"]


# ===========================================================================
# Warm leases (docs/WARMING.md "Product: a lease, held by the client")
#
# The property under test is not "warmth" but *release*: a client that crashes,
# sleeps, or walks away must stop costing money without doing anything. Most of
# what follows is really about the lease lapsing on its own.
# ===========================================================================
import asyncio  # noqa: E402

from hvym_img_tools import warm as warm_mod  # noqa: E402


class FakeRunPod:
    """Counts keepalive jobs and serves worker counts, so tests assert on what
    the pool actually sent rather than on wall-clock warmth."""

    def __init__(self, ready: int = 0, running: int = 0, initializing: int = 0):
        self.ready = ready
        self.running = running
        self.initializing = initializing
        self.warm_jobs = 0
        self.other_jobs = 0

    def install(self, monkeypatch):
        up = self

        class _Client:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers=None, json=None):
                tool = ((json or {}).get("input") or {}).get("tool")
                if tool == warm_mod.WARM_TOOL:
                    up.warm_jobs += 1
                else:
                    up.other_jobs += 1
                return httpx.Response(
                    200,
                    json={"status": "COMPLETED", "output": {"warm": True, "elapsed": 0.001}},
                    request=httpx.Request("POST", url),
                )

            async def get(self, url, headers=None):
                return httpx.Response(
                    200,
                    json={"workers": {"ready": up.ready, "running": up.running,
                                      "idle": 0, "initializing": up.initializing,
                                      "throttled": 0}},
                    request=httpx.Request("GET", url),
                )

        monkeypatch.setattr(warm_mod.httpx, "AsyncClient", _Client)
        return up


class Clock:
    """Controllable monotonic source, so lease expiry is exact, not timed."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _pool(clock, **kw):
    kw.setdefault("lease_ttl_s", 60.0)
    kw.setdefault("ping_interval_s", 0.01)
    return warm_mod.WarmPool("runpod-key", "ep123", clock=clock, **kw)


# --- the wire contract between proxy and worker ----------------------------

def test_warm_sentinel_matches_the_worker():
    """The literal is duplicated so the worker needs no HTTP stack. If the two
    ever drift, keepalives silently run the real pipeline instead."""
    from hvym_img_tools.serverless import WARM_TOOL as worker_side

    assert warm_mod.WARM_TOOL == worker_side


def test_worker_short_circuits_a_warm_job():
    from hvym_img_tools.serverless import handler

    out = handler({"input": {"tool": warm_mod.WARM_TOOL}})
    assert out["warm"] is True
    assert "error" not in out
    assert "data" not in out          # no pipeline ran


# --- auth ------------------------------------------------------------------

def test_warm_requires_the_scoped_key(env, monkeypatch):
    FakeRunPod().install(monkeypatch)
    client = _client()
    assert client.post("/warm").status_code == 401
    assert client.post("/warm", headers={"X-API-Key": "nope"}).status_code == 401
    assert client.request("DELETE", "/warm", json={"lease_id": "x"}).status_code == 401


def test_warm_status_is_open_for_the_indicator(env, monkeypatch):
    FakeRunPod(ready=1).install(monkeypatch)
    resp = _client().get("/warm")
    assert resp.status_code == 200
    assert resp.json()["state"] == "warm"


def test_warm_never_leaks_the_runpod_key(env, monkeypatch):
    FakeRunPod(ready=1).install(monkeypatch)
    client = _client()
    for body in (client.get("/warm").text,
                 client.post("/warm", headers={"X-API-Key": GOOD}).text):
        assert RUNPOD_KEY not in body
        assert "api.runpod.ai" not in body


def test_warm_acquire_over_http_returns_the_contract(env, monkeypatch):
    FakeRunPod(ready=1).install(monkeypatch)
    body = _client().post("/warm", headers={"X-API-Key": GOOD}).json()
    for field in ("lease_id", "state", "ready", "elapsed_s",
                  "expires_at", "lease_ttl_s", "renew_within_s"):
        assert field in body, field
    assert body["renew_within_s"] < body["lease_ttl_s"]


def test_delete_without_a_lease_id_is_422(env, monkeypatch):
    FakeRunPod().install(monkeypatch)
    resp = _client().request("DELETE", "/warm", headers={"X-API-Key": GOOD}, json={})
    assert resp.status_code == 422


# --- lease lifecycle -------------------------------------------------------

def test_acquire_extend_release(monkeypatch):
    fake = FakeRunPod(ready=0).install(monkeypatch)
    clock = Clock()

    async def scenario():
        pool = _pool(clock)
        first = await pool.acquire()
        assert first["state"] == "warming"      # lease held, worker not ready
        assert first["ready"] is False
        assert first["active_leases"] == 1
        lease_id = first["lease_id"]

        clock.advance(30)
        again = await pool.acquire(lease_id)
        assert again["lease_id"] == lease_id     # extended, not a second lease
        assert again["active_leases"] == 1
        assert pool._leases[lease_id].renewals == 1
        assert again["elapsed_s"] == 30.0        # elapsed tracks the session

        fake.ready = 1
        clock.advance(warm_mod.HEALTH_CACHE_S + 1)   # GET /warm caches health
        assert (await pool.status())["state"] == "warm"

        released = await pool.release(lease_id)
        assert released["active_leases"] == 0
        await pool.shutdown()

    asyncio.run(scenario())


def test_two_clients_share_one_worker(monkeypatch):
    """Refcounting: the second client must not double-pay, and the FIRST one
    leaving must not cut warmth out from under the second."""
    FakeRunPod(ready=1).install(monkeypatch)
    clock = Clock()

    async def scenario():
        pool = _pool(clock)
        await pool.acquire("inkternity-a", label="seat-a")
        await pool.acquire("inkternity-b", label="seat-b")
        assert (await pool.status())["active_leases"] == 2

        await pool.release("inkternity-a")
        assert (await pool.status())["active_leases"] == 1
        assert pool.loop_running, "must stay warm while B still holds a lease"

        await pool.release("inkternity-b")
        assert (await pool.status())["active_leases"] == 0
        await pool.shutdown()

    asyncio.run(scenario())


def test_release_is_idempotent(monkeypatch):
    FakeRunPod().install(monkeypatch)
    clock = Clock()

    async def scenario():
        pool = _pool(clock)
        await pool.acquire("only")
        assert (await pool.release("only"))["active_leases"] == 0
        # a retried DELETE after a dropped response must not error
        assert (await pool.release("only"))["active_leases"] == 0
        assert (await pool.release("never-existed"))["active_leases"] == 0
        await pool.shutdown()

    asyncio.run(scenario())


def test_lease_expires_without_the_client_doing_anything(monkeypatch):
    """The whole point: a crashed or sleeping client stops billing by itself."""
    FakeRunPod().install(monkeypatch)
    clock = Clock()

    async def scenario():
        pool = _pool(clock, lease_ttl_s=60.0)
        await pool.acquire("abandoned")
        assert (await pool.status())["active_leases"] == 1

        clock.advance(59)
        assert (await pool.status())["active_leases"] == 1, "survives a missed renewal"

        clock.advance(2)
        status = await pool.status()
        assert status["active_leases"] == 0
        assert status["state"] == "cold"
        assert status["expires_at"] is None
        await pool.shutdown()

    asyncio.run(scenario())


# --- the keepalive loop ----------------------------------------------------

def test_keepalive_fires_warm_jobs_and_stops_when_the_lease_lapses(monkeypatch):
    fake = FakeRunPod(ready=1).install(monkeypatch)
    clock = Clock()

    async def scenario():
        pool = _pool(clock, ping_interval_s=0.01)
        await pool.acquire("held")
        assert pool.loop_running

        await asyncio.sleep(0.08)
        assert fake.warm_jobs >= 1, "a held lease must keep pinging the worker"
        assert fake.other_jobs == 0, "keepalives must never run the real pipeline"

        clock.advance(120)                     # lease lapses; client is gone
        for _ in range(200):
            if not pool.loop_running:
                break
            await asyncio.sleep(0.01)

        assert not pool.loop_running, "loop must stop so the worker can sleep"
        fired = fake.warm_jobs
        await asyncio.sleep(0.05)
        assert fake.warm_jobs == fired, "no pings after the lease lapsed"
        await pool.shutdown()

    asyncio.run(scenario())


def test_tick_reports_idle_and_pings_only_while_leased(monkeypatch):
    fake = FakeRunPod(ready=1).install(monkeypatch)
    clock = Clock()

    async def scenario():
        pool = _pool(clock)
        await pool.acquire("x")
        await pool.shutdown()                  # drive ticks by hand

        assert await pool._tick() is True
        assert fake.warm_jobs == 1

        clock.advance(120)
        assert await pool._tick() is False     # idle -> loop would exit
        assert fake.warm_jobs == 1             # and it did not ping

    asyncio.run(scenario())


def test_ping_interval_keeps_margin_under_the_idle_timeout():
    """idleTimeout is 10s, and the worker's idle clock restarts when a job
    COMPLETES -- so the real gap is the sleep plus a RunPod round trip (~1s) plus
    the health check. A cadence merely "under 10" is not enough; it needs slack."""
    assert warm_mod.PING_INTERVAL_S <= 7, "leave room for round-trip latency"
    assert warm_mod.RENEW_WITHIN_S < warm_mod.LEASE_TTL_S / 2, "tolerate a missed poll"


# --- state is measured, never guessed --------------------------------------

def test_state_comes_from_real_worker_readiness(monkeypatch):
    fake = FakeRunPod(ready=0).install(monkeypatch)
    clock = Clock()

    async def scenario():
        pool = _pool(clock)
        assert (await pool.status())["state"] == "cold"

        await pool.acquire("a")
        status = await pool.status()
        assert status["state"] == "warming"    # leased but not ready
        assert status["ready"] is False

        # Readiness is cached briefly so a UI poll does not re-ask RunPod every
        # time; a real client polling every ~20s always sees past it.
        fake.ready = 2
        assert (await pool.status())["state"] == "warming", "cached within the window"
        clock.advance(warm_mod.HEALTH_CACHE_S + 1)
        status = await pool.status()
        assert status["state"] == "warm"
        assert status["workers_ready"] == 2
        await pool.shutdown()

    asyncio.run(scenario())


# --- keepalives must yield to real work ------------------------------------
# Found the hard way against the live endpoint: a keepalive firing alongside a
# real request let RunPod dispatch that request to a SECOND, cold worker. The
# handler itself took 2.681s; the client waited 137s. A tool request is already
# a job, so it resets idleTimeout on its own and a ping beside it is contention.

def test_no_keepalive_while_a_real_request_is_in_flight(monkeypatch):
    fake = FakeRunPod(ready=1).install(monkeypatch)
    clock = Clock()

    async def scenario():
        pool = _pool(clock, ping_interval_s=6.0)
        await pool.acquire("x")
        await pool.shutdown()                 # drive ticks by hand

        pool.request_started()
        clock.advance(2)
        assert await pool._tick() is True     # lease still held...
        assert fake.warm_jobs == 0            # ...but no competing ping

        pool.request_finished()
        assert await pool._tick() is True
        assert fake.warm_jobs == 0, "still too soon after the request finished"

        clock.advance(pool.ping_interval_s + 1)
        assert await pool._tick() is True
        assert fake.warm_jobs == 1            # idle again -> ping resumes

    asyncio.run(scenario())


def test_a_merely_queued_request_does_not_silence_the_keepalive(monkeypatch):
    """The bug this exists to prevent: a request RunPod has only QUEUED leaves
    the worker idle. Staying quiet for it let the worker sleep, and the queued
    job then waited out a fresh cold start -- a 0.024s cache hit measured at
    199s wall. Suppression must therefore be time-bounded."""
    fake = FakeRunPod(ready=1).install(monkeypatch)
    clock = Clock()

    async def scenario():
        pool = _pool(clock, ping_interval_s=6.0, lease_ttl_s=600.0)
        await pool.acquire("x")
        await pool.shutdown()

        pool.request_started()                # ...and it sits in RunPod's queue
        clock.advance(3)
        assert await pool._tick() is True
        assert fake.warm_jobs == 0, "brief overlap is still suppressed"

        clock.advance(30)                     # still not served
        assert await pool._tick() is True
        assert fake.warm_jobs >= 1, "a stalled request must not let the worker sleep"

    asyncio.run(scenario())


def test_overlapping_requests_keep_the_ping_suppressed(monkeypatch):
    fake = FakeRunPod(ready=1).install(monkeypatch)
    clock = Clock()

    async def scenario():
        pool = _pool(clock, ping_interval_s=6.0)
        await pool.acquire("x")
        await pool.shutdown()

        pool.request_started()
        pool.request_started()                # two artists at once
        pool.request_finished()               # one finishes
        clock.advance(2)                      # inside the suppression window
        assert await pool._tick() is True
        assert fake.warm_jobs == 0, "one request still in flight"

        pool.request_finished()
        clock.advance(pool.ping_interval_s + 1)
        assert await pool._tick() is True
        assert fake.warm_jobs == 1

    asyncio.run(scenario())


def test_proxy_notifies_the_pool_around_a_request(env, monkeypatch):
    """The counter must return to zero even when the upstream call fails, or
    keepalives stay suppressed for the rest of the lease."""
    FakeUpstream().install(monkeypatch, raises=httpx.ConnectError("boom"))
    app = proxy_mod.create_app()
    pool = app.state.warm_pool
    resp = TestClient(app).post(
        "/tools/reangle",
        files={"image": ("a.png", b"x", "image/png")},
        headers={"X-API-Key": GOOD},
    )
    assert resp.status_code == 502
    assert pool._inflight == 0, "a failed request must still release the ping suppression"


# --- the direct-server no-op ----------------------------------------------

def test_direct_server_warm_is_a_harmless_noop(monkeypatch):
    """A persistent box is always warm, so the client ships one code path."""
    monkeypatch.setenv("HVYM_DEVICE", "cpu")
    from hvym_img_tools.core.server import create_app as server_app

    client = TestClient(server_app())
    body = client.get("/warm").json()
    assert body["state"] == "warm"
    assert body["ready"] is True
    assert body["no_op"] is True
    assert client.post("/warm", json={"lease_id": "abc"}).json()["lease_id"] == "abc"


def test_a_busy_worker_still_counts_as_warm(monkeypatch):
    """Found on the live deployment. A worker EXECUTING a job reports `running`,
    not `ready`. Judging warmth by `ready` alone meant our own 6s keepalive kept
    the worker permanently busy from a well-connected proxy, so the state never
    left "warming" -- while the endpoint reported 41 completed jobs. Warm means a
    worker is up with models resident: ready, idle or running."""
    fake = FakeRunPod(ready=0, running=1).install(monkeypatch)
    clock = Clock()

    async def scenario():
        pool = _pool(clock)
        await pool.acquire("busy")
        status = await pool.status()
        assert status["state"] == "warm", "a running worker is warm, not warming"
        assert status["ready"] is True

        # Initializing is NOT warm -- it cannot serve without a wait.
        fake.running = 0
        fake.initializing = 1
        clock.advance(warm_mod.HEALTH_CACHE_S + 1)
        assert (await pool.status())["state"] == "warming"
        await pool.shutdown()

    asyncio.run(scenario())


# --- per-tool endpoint routing ---------------------------------------------
# Tools live on separate serverless endpoints (docs/tools/mesh.md §5), so the
# proxy has to route by tool while staying backward compatible with a
# deployment that only sets a single RUNPOD_ENDPOINT_ID.

def test_routes_each_tool_to_its_own_endpoint(env, monkeypatch):
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID_MESH", "mesh-ep")
    up = FakeUpstream().install(monkeypatch)
    client = _client()

    client.post("/tools/mesh", files={"image": ("a.png", b"x", "image/png")},
                headers={"X-API-Key": GOOD})
    assert up.seen["url"].endswith("/mesh-ep/runsync")

    client.post("/tools/reangle", files={"image": ("a.png", b"x", "image/png")},
                headers={"X-API-Key": GOOD})
    assert up.seen["url"].endswith("/ep123/runsync"), "reangle must keep the default"


def test_a_single_endpoint_still_works(env, monkeypatch):
    """An existing deployment sets only RUNPOD_ENDPOINT_ID and must not break."""
    up = FakeUpstream().install(monkeypatch)
    _client().post("/tools/mesh", files={"image": ("a.png", b"x", "image/png")},
                   headers={"X-API-Key": GOOD})
    assert up.seen["url"].endswith("/ep123/runsync")


def test_cache_key_is_surfaced_for_the_library(env, monkeypatch):
    FakeUpstream(payload={
        "status": "COMPLETED",
        "output": {"data": base64.b64encode(GLB).decode(),
                   "media_type": "model/gltf-binary", "cache_key": "abc123"},
    }).install(monkeypatch)
    resp = _client().post("/tools/mesh", files={"image": ("a.png", b"x", "image/png")},
                          headers={"X-API-Key": GOOD})
    assert resp.headers["x-cache-key"] == "abc123"


def test_warm_targets_the_named_tool(env, monkeypatch):
    """Warming both endpoints would double the bill for no benefit."""
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID_MESH", "mesh-ep")
    FakeRunPod(ready=1).install(monkeypatch)
    app = proxy_mod.create_app()
    client = TestClient(app)

    client.post("/warm", headers={"X-API-Key": GOOD}, json={"lease_id": "a", "tool": "mesh"})
    client.post("/warm", headers={"X-API-Key": GOOD}, json={"lease_id": "b", "tool": "reangle"})

    body = client.get("/warm", params={"tool": "mesh"}).json()
    assert body["active_leases"] == 1, "the mesh lease must not count reangle's"


def test_cache_hit_still_reports_the_tool_version():
    """A library keys assets on the version that produced them.

    X-Tool-Version came only from the worker's MISS branch, so the header
    vanished on exactly the responses that are fast -- the same asset returned
    different metadata depending on cache state. Verified against the live
    endpoint: MISS carried x-tool-version: 0.1.0, HIT carried none.
    """
    import inspect

    import hvym_img_tools.serverless as sl

    src = inspect.getsource(sl.run_tool if hasattr(sl, "run_tool") else sl.handler)
    hit = src[src.index('"cached": True'):]
    hit = hit[: hit.index("try:")] if "try:" in hit else hit
    assert "tool_version" in hit, "cache hits must report the tool version too"
