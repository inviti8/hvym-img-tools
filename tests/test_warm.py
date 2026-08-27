"""Keepalive timing invariants.

The lease exists to stop a worker sleeping. These pin the arithmetic that has
to hold for that to work at all -- the rest of the pool's behaviour is covered
in test_proxy.py.
"""
# --- the keepalive must outlive the endpoint's idle timeout ------------------

def test_ping_interval_stays_under_the_endpoint_idle_timeout():
    """The invariant the whole lease rests on.

    If we can go longer than idleTimeout without pinging, the worker we are
    paying to keep awake sleeps anyway, and the next request pays a cold start.
    """
    from hvym_img_tools.warm import IDLE_TIMEOUT_S, PING_INTERVAL_S

    assert PING_INTERVAL_S < IDLE_TIMEOUT_S, (
        f"ping every {PING_INTERVAL_S}s cannot keep a worker that sleeps after "
        f"{IDLE_TIMEOUT_S}s awake"
    )


def test_keepalive_continues_while_a_request_is_queued():
    """A queued request does NOT keep the worker awake -- so we must.

    Regression: pings were suppressed for 2 ping intervals (12s) after a
    request started, while the endpoint sleeps at 10s. A warm, leased endpoint
    queued a real request for ~3 minutes because the worker went to sleep
    inside that 2-second gap and the job fell back to a cold start.
    """
    from hvym_img_tools.warm import IDLE_TIMEOUT_S, WarmPool

    clock = [1000.0]
    pool = WarmPool("k", "ep", clock=lambda: clock[0])

    pool.request_started()
    # Step forward to just past the point where the worker would sleep.
    clock[0] += IDLE_TIMEOUT_S + 0.1
    assert pool._ping_needed(), (
        "must keep pinging while a request is in flight -- it may be queued, "
        "not executing, and the worker sleeps on its own"
    )
