"""Client-held warm leases (docs/WARMING.md, "Product: a lease, held by the client").

**Why a lease and not a switch.** A switch is state the client *asserts*: if
Inkternity crashes, the laptop sleeps, or the artist walks away, it stays on and
keeps billing. A lease is state the client must keep *re-asserting* — silence
means release. The failure mode of a lease is "goes cold", which is free. With
paying artists on the other end, only one of those is shippable.

`scripts/warm.py` is the *operator* switch and deliberately uses `workersMin`,
because persistence past process death is wanted there. Here it is exactly wrong:
if the proxy dies, `workersMin` would leave the endpoint warm forever. So warmth
is held by keepalive jobs fired from this process — when it stops, the worker
sleeps on its own.

**Two clocks, deliberately decoupled:**

* the *lease* clock (`lease_ttl_s`, default 60) is the client's. Each `POST /warm`
  pushes the deadline out; the client is told to renew every `renew_within_s`
  (default 20), so one or two dropped renewals are survivable.
* the *ping* clock (`ping_interval_s`, default 6) is ours, and must stay under the
  endpoint's `idleTimeout` (10 s) or the worker sleeps between pings.

Nothing here ever returns the RunPod account key, or any URL containing it.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx

log = logging.getLogger(__name__)

RUNPOD_BASE = "https://api.runpod.ai/v2"

#: How long a single renewal buys. Generous relative to the renew cadence so a
#: dropped poll does not drop the worker.
LEASE_TTL_S = float(os.environ.get("HVYM_LEASE_TTL_S", "60"))

#: What we tell the client its renewal cadence should be. Must be comfortably
#: under LEASE_TTL_S -- at these defaults two consecutive misses are tolerated.
RENEW_WITHIN_S = float(os.environ.get("HVYM_LEASE_RENEW_WITHIN_S", "20"))

#: Must stay BELOW the endpoint's idleTimeout (10 s) *with margin*. The worker's
#: idle clock restarts when a job COMPLETES, so the real gap is this sleep plus a
#: round trip to RunPod (~0.7-1.3 s measured) plus the health check. At 8 s that
#: totals ~9.7 s against a 10 s timeout -- one latency spike and the worker sleeps,
#: making the next renewal pay a full cold start. 6 s leaves ~2.5 s of headroom,
#: and the extra pings are free: a __warm__ job is a no-op on a worker already
#: being paid for.
PING_INTERVAL_S = float(os.environ.get("HVYM_WARM_PING_INTERVAL_S", "6"))

#: A keepalive uses /runsync so a cold start blocks it rather than letting us
#: queue a ping every few seconds for four minutes. This bounds that block.
PING_TIMEOUT_S = float(os.environ.get("HVYM_WARM_PING_TIMEOUT_S", "90"))

#: GET /warm is a UI poll; don't re-ask RunPod for worker counts on every call.
HEALTH_CACHE_S = 3.0

#: The job the worker short-circuits on (see serverless.py). Its only purpose is
#: to be a completed job, which is what resets idleTimeout.
WARM_TOOL = "__warm__"


def _iso(seconds_from_now: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=max(0.0, seconds_from_now))).isoformat()


@dataclass
class Lease:
    """One client's hold on the warm worker.

    Shaped for metering from the start: WARMING.md makes lease duration the
    billable unit, and refcounting decides who pays when sessions overlap.
    Retrofitting attribution onto an unmetered warm path is much harder than
    carrying `label`/`renewals`/`acquired_wall` from the first commit.
    """

    lease_id: str
    label: str = ""
    acquired_at: float = 0.0          # monotonic, for arithmetic
    expires_at: float = 0.0           # monotonic
    acquired_wall: float = field(default_factory=time.time)  # wall, for records
    renewals: int = 0

    def held_s(self, now: float) -> float:
        return max(0.0, now - self.acquired_at)


class WarmPool:
    """Refcounted warm leases plus the keepalive loop that honours them."""

    def __init__(
        self,
        runpod_key: str,
        endpoint_id: str,
        *,
        lease_ttl_s: float = LEASE_TTL_S,
        ping_interval_s: float = PING_INTERVAL_S,
        renew_within_s: float = RENEW_WITHIN_S,
        clock=time.monotonic,
    ) -> None:
        self._key = runpod_key
        self._endpoint_id = endpoint_id
        self.lease_ttl_s = lease_ttl_s
        self.ping_interval_s = ping_interval_s
        self.renew_within_s = renew_within_s
        self._clock = clock

        self._leases: dict[str, Lease] = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

        #: When the current warm *session* began -- i.e. when the pool last went
        #: from zero leases to one. Drives elapsed_s so the client can say
        #: "warming, up to ~4 min" without us inventing an ETA.
        self._session_started: float | None = None

        self._workers_ready = 0
        self._health_checked_at: float | None = None
        self._pings = 0

        #: Real work in flight, and when it last finished. A tool request is
        #: itself a job, so it resets idleTimeout exactly like a ping does --
        #: pinging alongside it is pure contention. Measured: a keepalive firing
        #: next to a real request let RunPod dispatch the request to a SECOND,
        #: cold worker, turning a 2.7s job into a 137s wait.
        self._inflight = 0
        self._last_activity: float | None = None
        self._inflight_since: float | None = None

    # ---------------------------------------------------------------- helpers
    @property
    def configured(self) -> bool:
        return bool(self._key and self._endpoint_id)

    def _prune(self) -> list[Lease]:
        """Drop lapsed leases. Called from every read as well as the loop, so a
        stalled loop can never make GET /warm report a lease that has expired."""
        now = self._clock()
        dead = [lease for lease in self._leases.values() if lease.expires_at <= now]
        for lease in dead:
            del self._leases[lease.lease_id]
            # The metering hook: this line is the billable record.
            log.info(
                "warm lease released (expired) id=%s label=%s held=%.1fs renewals=%d",
                lease.lease_id[:8], lease.label or "-", lease.held_s(now), lease.renewals,
            )
        if not self._leases:
            self._session_started = None
        return dead

    def _state(self) -> str:
        """Never a guess. `warm` comes only from workers.ready > 0."""
        if self._workers_ready > 0:
            return "warm"
        return "warming" if self._leases else "cold"

    def _elapsed_s(self) -> float:
        if self._session_started is None:
            return 0.0
        return max(0.0, self._clock() - self._session_started)

    def _next_expiry_iso(self) -> str | None:
        if not self._leases:
            return None
        latest = max(lease.expires_at for lease in self._leases.values())
        return _iso(latest - self._clock())

    # ------------------------------------------------------------ RunPod I/O
    async def _get_health(self, force: bool = False) -> int:
        """workers.ready from the endpoint's own health, lightly cached."""
        now = self._clock()
        if (
            not force
            and self._health_checked_at is not None
            and now - self._health_checked_at < HEALTH_CACHE_S
        ):
            return self._workers_ready
        if not self.configured:
            return 0
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{RUNPOD_BASE}/{self._endpoint_id}/health",
                    headers={"Authorization": f"Bearer {self._key}"},
                )
            if resp.status_code < 400:
                workers = (resp.json() or {}).get("workers") or {}
                self._workers_ready = int(workers.get("ready") or 0)
        except (httpx.HTTPError, ValueError):
            # A health blip must not change the answer to "is a lease held".
            log.debug("warm: health check failed", exc_info=True)
        self._health_checked_at = now
        return self._workers_ready

    async def _ping(self) -> None:
        """Fire one keepalive job. Uses /runsync deliberately: during a cold start
        it blocks instead of returning instantly, which stops us queueing thirty
        no-op jobs across four minutes of container pull."""
        if not self.configured:
            return
        try:
            async with httpx.AsyncClient(timeout=PING_TIMEOUT_S) as client:
                await client.post(
                    f"{RUNPOD_BASE}/{self._endpoint_id}/runsync",
                    headers={"Authorization": f"Bearer {self._key}"},
                    json={"input": {"tool": WARM_TOOL}},
                )
            self._pings += 1
        except httpx.HTTPError:
            # Never surface upstream detail: it can carry the endpoint URL.
            log.debug("warm: keepalive ping failed", exc_info=True)

    # -------------------------------------------------------------- activity
    def request_started(self) -> None:
        """Told by the proxy when a real tool request begins."""
        if self._inflight == 0:
            self._inflight_since = self._clock()
        self._inflight += 1
        self._last_activity = self._clock()

    def request_finished(self) -> None:
        self._inflight = max(0, self._inflight - 1)
        self._last_activity = self._clock()
        if self._inflight == 0:
            self._inflight_since = None

    def _ping_needed(self) -> bool:
        """A ping is only worth sending when nothing else is keeping the worker
        awake.

        "In flight" is not the same as "executing". A request that RunPod has
        merely QUEUED leaves the worker idle, so staying quiet for it lets the
        worker sleep -- and the queued job then waits out a fresh cold start.
        Measured: suppressing for the whole of a queued request turned a 0.024s
        cache hit into a 199s wall. So the suppression is time-bounded: if a
        request has not completed within a couple of ping intervals it is not
        being served promptly, and keeping the worker alive matters more than
        avoiding contention with it.
        """
        now = self._clock()
        if self._inflight and self._inflight_since is not None:
            if (now - self._inflight_since) < self.ping_interval_s * 2:
                return False
            return True          # queued, not executing -- keep the worker up
        if self._inflight:
            return False
        if self._last_activity is None:
            return True
        return (now - self._last_activity) >= self.ping_interval_s

    # ----------------------------------------------------------------- loop
    async def _tick(self) -> bool:
        """One keepalive iteration. Returns False when the pool has gone idle.

        Split out from the loop so tests can drive it deterministically rather
        than sleeping on wall-clock intervals.
        """
        async with self._lock:
            self._prune()
            alive = bool(self._leases)
        if not alive:
            return False
        if self._ping_needed():
            await self._ping()
        await self._get_health(force=True)
        return True

    async def _run(self) -> None:
        log.info("warm: keepalive loop started (every %.0fs)", self.ping_interval_s)
        try:
            while await self._tick():
                await asyncio.sleep(self.ping_interval_s)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        finally:
            log.info("warm: keepalive loop stopped; worker sleeps on its own")

    def _ensure_loop(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    @property
    def loop_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def shutdown(self) -> None:  # pragma: no cover - lifespan path
        task, self._task = self._task, None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # ------------------------------------------------------------- public API
    async def acquire(self, lease_id: str | None = None, label: str = "") -> dict:
        now = self._clock()
        async with self._lock:
            self._prune()
            lease = self._leases.get(lease_id) if lease_id else None
            if lease is None:
                lease = Lease(
                    lease_id=lease_id or uuid.uuid4().hex,
                    label=label,
                    acquired_at=now,
                    expires_at=now + self.lease_ttl_s,
                )
                self._leases[lease.lease_id] = lease
                if self._session_started is None:
                    self._session_started = now
                log.info(
                    "warm lease acquired id=%s label=%s (active=%d)",
                    lease.lease_id[:8], label or "-", len(self._leases),
                )
            else:
                lease.expires_at = now + self.lease_ttl_s
                lease.renewals += 1
                if label:
                    lease.label = label
            self._ensure_loop()

        await self._get_health()
        return self._view(lease)

    async def release(self, lease_id: str) -> dict:
        """Idempotent: releasing an unknown or already-expired lease is a no-op
        that still reports the truth, so a client retrying a DELETE after a
        dropped response does not get an error it cannot act on."""
        async with self._lock:
            self._prune()
            lease = self._leases.pop(lease_id, None)
            if lease is not None:
                now = self._clock()
                log.info(
                    "warm lease released id=%s label=%s held=%.1fs renewals=%d (active=%d)",
                    lease.lease_id[:8], lease.label or "-", lease.held_s(now),
                    lease.renewals, len(self._leases),
                )
            if not self._leases:
                self._session_started = None
        return await self.status()

    async def status(self) -> dict:
        async with self._lock:
            self._prune()
        await self._get_health()
        return {
            "state": self._state(),
            "ready": self._workers_ready > 0,
            "workers_ready": self._workers_ready,
            "active_leases": len(self._leases),
            "elapsed_s": round(self._elapsed_s(), 1),
            "expires_at": self._next_expiry_iso(),
            "lease_ttl_s": self.lease_ttl_s,
            "renew_within_s": self.renew_within_s,
        }

    def _view(self, lease: Lease) -> dict:
        return {
            "lease_id": lease.lease_id,
            "state": self._state(),
            "ready": self._workers_ready > 0,
            "elapsed_s": round(self._elapsed_s(), 1),
            "expires_at": _iso(lease.expires_at - self._clock()),
            "lease_ttl_s": self.lease_ttl_s,
            "renew_within_s": self.renew_within_s,
            "active_leases": len(self._leases),
        }


#: What a deployment with no scale-to-zero reports. A persistent pod running
#: core.server is always warm, so /warm is a truthful no-op there -- which lets
#: Inkternity ship one code path regardless of how the service is deployed.
def always_warm_view(lease_id: str | None = None) -> dict:
    return {
        "lease_id": lease_id or uuid.uuid4().hex,
        "state": "warm",
        "ready": True,
        "workers_ready": 1,
        "active_leases": 0,
        "elapsed_s": 0.0,
        "expires_at": None,
        "lease_ttl_s": 0.0,
        "renew_within_s": 0.0,
        "no_op": True,
    }
