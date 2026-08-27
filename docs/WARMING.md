# WARMING.md — the demo switch vs. the product lease

A cold worker takes **up to ~260 s** to serve its first request (BENCHMARK.md §6b).
Warming hides that, and warming costs money. Those two facts pull in opposite
directions, and the right resolution is **different for the demo than for the
shipped product** — so they are built differently on purpose.

## The number that drives all of this

| | |
|---|---|
| Warm worker, idle | **~$1.12/hour** (24 GB flex tier) |
| Same money as | **~2,000 images** of on-demand compute |
| An 8-hour day left warm | ~$9 |
| A working month left warm | ~$200 — the persistent-pod cost serverless exists to avoid |

Warming does not merely add cost, it **inverts the cost model**. On-demand,
1,000 drawings/month is ~$0.55 of compute. One forgotten switch over a long
weekend costs more than a year of that.

That is why the demo and the product diverge: the question is not *"how do we
warm?"* but **"who is trusted to turn it off, and what happens if they don't?"**

## Demo: a plain switch, operated by hand

`scripts/warm.py` — no auto-expiry, no timer, no client involvement.

```sh
uv run python scripts/warm.py on       # start a worker, keep it up
uv run python scripts/warm.py wait     # block until it can serve (~1-4 min)
uv run python scripts/warm.py status   # up? how long? what has it cost?
uv run python scripts/warm.py off      # release it
```

Turn it on before a demo, `wait` for ready, run the demo with no cold start, turn
it off after. One operator, one endpoint, deliberate action at both ends.

**Implemented with `workersMin` 0↔1.** A switch should not need a process
babysitting it: RunPod's endpoint config is the one piece of state guaranteed to
outlive the script, survive a reboot, and stay put until explicitly changed.
Persistence is the *desired* semantic here.

**The safety story is you.** `status` reports elapsed time and estimated spend
because nothing else will — there is no timeout to catch a forgotten switch.
That is an acceptable trade for a tool one person drives by hand, and an
unacceptable one for anything else.

## Product: a lease, held by the client

**Implemented** — `POST/GET/DELETE /warm` on the proxy (`hvym_img_tools/warm.py`),
with the worker-side `__warm__` short-circuit in `serverless.py`. Client contract
in [CLIENT.md](CLIENT.md#warming).


The moment a *client* asks for warmth, the switch model breaks — not because the
mechanism differs, but because the failure mode does.

> A **switch** is state the client *asserts*. If Inkternity crashes, the network
> drops, the laptop sleeps, or the artist walks away, it is stuck ON and billing.
>
> A **lease** is state the client must keep *re-asserting*. Silence means release.

The failure mode of a lease is "goes cold" — free. The failure mode of a switch
in client hands is "bills all weekend". With paying artists on the other end,
only one of those is shippable.

### Shape

Three endpoints on the proxy, which already holds the RunPod key:

| | |
|---|---|
| `POST /warm` | acquire or extend; returns `{lease_id, state, ready, elapsed_s, expires_at, lease_ttl_s, renew_within_s}` |
| `GET /warm` | current state, for UI not holding a lease (unauthenticated: it spends nothing and cannot start a worker) |
| `DELETE /warm` | explicit release when the artist flips it off |

Leases are refcounted, so two Inkternity instances behind one proxy share one
warm worker and do not pay twice. The last release stops the keepalive loop.

**Two clocks, deliberately decoupled.** The client's renewal cadence and our
keepalive cadence answer different questions and must not be tied together:

| | Default | Env | Bound by |
|---|---|---|---|
| Lease TTL | 60 s | `HVYM_LEASE_TTL_S` | how long a crashed client may keep billing |
| Renew within | 20 s | `HVYM_LEASE_RENEW_WITHIN_S` | must be well under the TTL, so a dropped poll is survivable |
| Keepalive ping | 6 s | `HVYM_WARM_PING_INTERVAL_S` | **must stay under the endpoint's `idleTimeout` (10 s)** or the worker sleeps between pings |

At these defaults two consecutive missed renewals are tolerated, and an
abandoned lease releases within ~60 s — after which the worker sleeps on its own
about 10 s later.

The keepalive uses `/runsync` rather than `/run` on purpose: during a cold start
it blocks instead of returning instantly, which stops the proxy queueing thirty
no-op jobs across four minutes of container pull.

**The renewal poll is the notification channel.** The client is already talking
to the proxy every ~20 s to hold its lease, so the response carries state and
expiry. No push, no WebSocket, no reconnect logic in a C++ client. If the poll
itself fails, the UI shows cold — which is correct, because it probably is.

**Mechanism: keepalive jobs, not `workersMin`.** Here the persistence that makes
`workersMin` right for the demo makes it wrong: if the proxy dies, the endpoint
stays warm forever. Cheap no-op jobs that reset `idleTimeout` stop when the proxy
stops, and the worker drops itself. The safe behaviour is the automatic one.

`serverless.py` short-circuits `__warm__` before any pipeline work — at one ping
every 8 s, running reconstruction would burn GPU seconds for the whole lease and
produce nothing anyone reads. The sentinel is duplicated in `warm.py` rather than
imported, so the worker takes no dependency on the proxy's HTTP stack; a test
asserts the two definitions agree.

### Match the mechanism to the meaning

| Semantic | Mechanism | Persists past process death | Failure mode |
|---|---|---|---|
| **Switch** (operator) | `workersMin` 0↔1 | yes — *wanted* | stays warm, you are watching |
| **Lease** (client) | keepalive jobs | no — *wanted* | goes cold, costs nothing |

### Billing implication

Active inference is the dominant cost in the paid product, so **artists pay for
warm time**, not per image. The lease is therefore also the metering boundary:
lease duration is the billable unit, and refcounting decides who is charged when
sessions overlap. Design the lease record with that in mind from the start —
retrofitting attribution onto an unmetered warm path is far harder.

The `Lease` dataclass therefore carries `label` (client-supplied attribution),
`acquired_wall`, `renewals` and a `held_s()` duration from the first commit, and
every acquire/release/expiry emits a log line with them. That is the metering
hook: no billing system yet, but nothing to reconstruct later either.

### Measured against the live endpoint

Endpoint `69j3vhp0el0wv0`, image `0.1.1`, lease renewed every 15 s:

| | |
|---|---|
| Cold ➜ `state: "warm"` | **26 s** |
| Request while leased | `X-Cache: MISS`, **2.527 s** of upstream work |
| Same, unleased/cold | ~50-260 s (BENCHMARK.md §6b) |
| Release ➜ `state: "cold"` | **1-2 s**, and the worker sleeps ~10 s later |

**`workersMax` matters more than it looks.** At 2, a request issued while a lease
was held got dispatched to a *second, throttled* worker and sat `IN_QUEUE` for
~137 s while a warm worker sat ready. EU-RO-1 has shown a persistent
`throttled: 1` all along. At 1 the lease and the request share the same box and
the queueing disappeared. `deploy_runpod.py` therefore defaults to 1; raise it
only when concurrent drawings genuinely need it.

**Wall-clock from a laptop is not the service's latency.** The same cache hit
(0.026 s of work) measured 4.5 s earlier in the day and 83 s later, purely
because this machine's uplink degraded — it carries ~745 KB up and ~665 KB down
per request. Judge warmth by `x-upstream-elapsed`, not by the stopwatch.

## Measured: mesh through nginx, 2026-08-27

One cold request for `/tools/mesh` end to end through `https://img.hvym.link`,
endpoint `km99b7mrj2f85r`, zero ready workers and `throttled: 1` at dispatch:

| | |
|---|---|
| Total wall clock | **547 s** |
| `X-Upstream-Elapsed` (real GPU work) | **8.699 s** |
| Everything else -- placement + cold boot | ~538 s |
| Result | 200, 360,504 B, valid `glTF` |

A warm repeat of the same sketch: 4.4-4.9 s wall, `X-Cache: HIT`,
`X-Upstream-Elapsed: 0.025` -- the volume-backed result cache answering.

Two things to take from it. **98% of a cold request is waiting for hardware**,
so tuning inference buys nothing next to holding a worker. And nginx's
`proxy_read_timeout` has to clear 547 s: at the old 300 s default this exact
request came back as a 504 HTML page after five minutes, having completed
successfully on RunPod the whole time. It is 900 s now.

**Don't read the queue from `/v2/{id}/health`.** During that run it reported
`inQueue: 1, completed: 15` well after the job had returned. It is the same
endpoint whose `workers.ready` once hid a warm worker from us. Trust the
client's stopwatch and `X-Upstream-Elapsed`; treat the health counts as a hint.

## Don't fake the ETA

Cold start is ~260 s onto a host without the image but ~48 s onto one that has
it, and neither the script nor the proxy can tell which it got. Show elapsed with
"up to ~4 min" and flip to ready on a real signal — `workers.ready > 0` from
`/v2/{id}/health`, which is what `warm.py wait` polls. A counted-down fake ETA
that expires while still warming reads as broken.

## The cheaper fix nobody pays hourly for

260 s is ~61% one 3.96 GB torch+CUDA layer. torch's wheels bundle cuDNN (706 MB),
cuBLAS, cuFFT, cuSOLVER and NCCL; we run single-GPU TripoSR plus onnxruntime-gpu,
which duplicates some of them. Trimming toward ~4 GB would put cold start near
~170 s. Fiddly and not transformative — but it is the only lever that helps every
user without an hourly bill, and it makes both designs above cheaper.

## Open: verify the rate against real billing

`USD_PER_SECOND` in `warm.py` (and the table above) is the published 24 GB flex
rate, not a measured one. Two things need confirming against `/billing/endpoints`
once a day of real usage has posted:

1. **The warm-hour rate.** Everything here scales off it.
2. **Whether idle time inside `idleTimeout` is billed.** If it is, an isolated
   on-demand request costs ~10 s of idle on top of ~1.9 s of work — making the
   real per-image figure closer to ~$0.004 than the $0.00054 of work-only time
   in BENCHMARK.md §2. Batched work amortizes it; scattered one-off drawings do
   not. This also sets how far `idleTimeout` can rise before it hurts the
   on-demand model.

```sh
uv run python -c "import json,urllib.request,os; \
os.environ.update(l.split('=',1) for l in open('.env') if '=' in l and not l.startswith('#')); \
print(json.dumps(json.load(urllib.request.urlopen(urllib.request.Request( \
  'https://rest.runpod.io/v1/billing/endpoints', \
  headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'].strip()}))),indent=1))"
```
