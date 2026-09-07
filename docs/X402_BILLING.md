# X402_BILLING.md — what the proxy must build to charge for warm time

**Status:** **built, shipped off.** Phases 2 and 3 are implemented in this repo —
`core/identity.py` (§2), `billing.py` (§3–§4), the routes in `proxy.py`, and
`tests/test_billing.py` (§7). Both enforcement flags default to `false`, so the
proxy today verifies whatever signatures arrive, logs what it *would* have
refused, and serves every request exactly as before. What remains is the
**cutover** row of §6 — and Phase 0, which gates it. The client half (Phase 1) is
already shipped in Inkternity.
**Read first:** [WARMING.md](WARMING.md) (the lease this bills), [AUTH.md](AUTH.md)
(the shared key this supersedes as the billing primitive — see its §Upgrade path,
steps 2 and 4: this doc *is* steps 2 and 4).
**Authoritative decision doc (client side):** `../infinipaint/docs/design/AI_BILLING_INTEGRATION.md`.
**Client wire contract (already implemented):** `../infinipaint/src/AI/RequestSigner.hpp`.

---

## 0. The one rule everything follows

**The inference endpoint is always payment-required. There is no free tier.** No live
paid window for an identity → no `/warm`, no `/tools/*`, full stop. An unfunded wallet
gets nothing.

That single decision (owner: Inkternity, `AI_BILLING_INTEGRATION.md` §0) is why this is
simpler than a normal billing system:

- **No portal identity, no accounts table.** Auth is a self-authenticating client-side
  wallet signature. We verify a signature and check a paid-through timestamp — nothing else.
- **No server-held balance, no ledger.** We do not custody funds. The artist pays P2P
  from their own Stellar wallet to ours; we verify the payment *landed on-chain* and extend
  a window. State is one timestamp per pubkey plus a consumed-tx set.
- **No fiat, no merchant-of-record, no on-ramp.** HEAVYMETA only ever *receives* crypto for
  its own GPU service.

Free-riding collapses into "spending a funded wallet": a cracked/cloned client still has to
pay per window from a wallet someone funded. The binary need not be secret.

---

## 1. What changes, in one picture

```
Inkternity (crypto rails ON)                Proxy (this repo)                 RunPod
  every /warm + /tools/* call ── X-Ink-Pubkey + X-Ink-Auth (ed25519) ─►  verify signature
  POST /warm (acquire/renew)  ───────────────────────────────────────►  paid window live?
        │                                                                  ├─ yes → WarmPool.acquire (as today)
        │                                                                  └─ no  → 402 { payment requirements }
  POST /warm/pay  ── X-Payment: <stellar tx hash> ───────────────────►  verify on Horizon → extend paid_through
  POST /tools/{name} ────────────────────────────────────────────────►  paid window live? → proxy to RunPod (as today)
  GET  /warm  (unchanged: open, status only, spends nothing)
```

Today (`proxy.py`) every `/warm` and `/tools/*` call is gated by one shared `X-API-Key`
(`core/auth.py`), and `Lease.label` is a client-supplied string (`warm.py`). After this work:

- **`X-API-Key` stays** as a coarse bot-gate / kill-switch, *not* as the billing identity.
  Do not remove it (the client still sends it; retire later).
- **Identity is a verified wallet pubkey**, from a per-request ed25519 signature.
- **`Lease.label` becomes the verified pubkey**, set by the proxy from the signature —
  never trusted from the client body.
- **Acquire/renew and tool calls require a live paid window** for that pubkey.

---

## 2. Verify the signed identity (mirror of the client)

Every `POST /warm`, `DELETE /warm`, `POST /warm/pay`, and `POST /tools/{name}` carries two
headers the client already sends (`RequestSigner.hpp`):

```
X-Ink-Pubkey: <G... Stellar strkey>
X-Ink-Auth:   <base64url(64-byte ed25519 signature)> "." <base64url(payload)>
```

`payload` is the **exact signed bytes**: compact JSON, no spaces, keys in this order
(alphabetical), and you MUST verify against these bytes as received — do not re-serialize:

```json
{"i":"ai-request","lid":"<lease_id|>","m":"<METHOD>","n":"<nonce-hex>","p":"<path>","t":<unix-seconds>,"tool":"<tool>"}
```

`X-Ink-Auth` is the same envelope shape as `../infinipaint/src/C2PA/WireToken` and Inkternity's
`Subscription/TokenVerifier`.

### Verification steps (all must pass, else `401`)
1. Split `X-Ink-Auth` on `.`; base64url-decode both halves (URL-safe, no padding). Signature
   must be 64 bytes; payload non-empty.
2. Decode `X-Ink-Pubkey` from Stellar strkey → 32-byte ed25519 public key (`stellar_sdk.StrKey.decode_ed25519_public_key`).
3. **Verify the ed25519 signature over the raw payload bytes** with that public key
   (`stellar_sdk.Keypair.from_public_key(...).verify(payload_bytes, sig)`, or PyNaCl).
4. Parse the payload JSON and bind it to the actual request:
   - `payload["i"] == "ai-request"`
   - `payload["m"]` == the HTTP method
   - `payload["p"]` == the request path **as routed to FastAPI** (no scheme/host/query;
     e.g. `/warm`, `/tools/mesh`). If deployed behind an nginx prefix, verify against the
     path the app sees.
   - `payload["tool"]` == the tool (for `/tools/{name}` it is `name`; for `/warm` it is the
     body's `tool`, default `reangle`)
   - `payload["lid"]` == the body `lease_id` (empty string on first acquire)
5. **Freshness:** `abs(now - payload["t"]) <= HVYM_SIGN_WINDOW_S` (default 120). Reject
   otherwise.
6. **Replay:** `payload["n"]` (32-hex nonce) must be unseen within the freshness window.
   Keep a small TTL set keyed by `(pubkey, n)`, TTL = `HVYM_SIGN_WINDOW_S`.

The verified `pubkey` (the strkey) is the **identity** for everything below. It is also the
metering label — set `Lease.label = pubkey`, ignoring any client-supplied `label`.

### Body is intentionally NOT signed
The `/tools/*` body is a curl-generated multipart the client cannot hash as one buffer, so
the signature covers method/path/tool/time/nonce, not the body. On the paid-only model the
exposure is bounded (a captured header spends the *payer's own* paid window, one nonce, once,
inside the freshness window). Do not build body-binding now; revisit only if it proves needed.

### Phased rollout (do not break today's client)
Two env flags, so verify can ship before it enforces:
- `HVYM_REQUIRE_SIGNED_IDENTITY` (default `false`): when false, verify-and-log only — a
  missing/invalid signature is recorded but the request proceeds on `X-API-Key` alone. Flip
  to `true` once telemetry shows real clients signing.
- `HVYM_REQUIRE_PAYMENT` (default `false`): gates §3–§4. When false, a verified identity with
  no paid window is still served (current free behaviour) — lets you land identity + payment
  verification before turning the meter on.

Ship order: verify (log-only) → require signed identity → require payment.

---

## 3. Paid-through window accounting (per pubkey)

Minimal durable state — **not** a ledger:

```
paid_through: dict[pubkey_strkey -> unix_ts]     # window end
consumed_tx:  set[stellar_tx_hash]               # idempotency
```

Persist both (survive a proxy restart — the artist paid). SQLite is enough; Redis if you
already run one. Do **not** hold balances.

- `is_paid(pubkey) := now < paid_through.get(pubkey, 0)`.
- **Gate (`HVYM_REQUIRE_PAYMENT=true`):** `POST /warm` acquire/renew and `POST /tools/{name}`
  require `is_paid(pubkey)`. If not → `402` with the challenge from §4.
- `GET /warm` stays **open and unauthenticated** — status only, spends nothing, starts no
  worker (unchanged from today; do not gate it).
- `DELETE /warm` requires a verified identity that **owns** the lease (the lease's label ==
  pubkey). Release never needs payment.

### Settle on grant, not on payment (§2.6, the "no warm, no charge" rule)
A window must not be consumed by a worker that never warmed (throttled/cold the artist waited
on but never got). So the window clock **starts when the worker first reaches `state=="warm"`
for that identity after payment**, not at payment receipt. Implement as a credit:

```
On verified payment:  paid_credit_s[pubkey] += window_s      # granted, not yet running
First time WarmPool reports warm for a held lease of pubkey:
                      paid_through[pubkey] = now + paid_credit_s[pubkey]; paid_credit_s = 0
```

`WarmPool` already knows warmth (`_state()` / `workers_ready`); wire the transition there.
Keep it simple: one credit outstanding at a time is fine for v1.

### Attribution ties into the existing lease record
`warm.py`'s `Lease` already carries `label`, `acquired_wall`, `renewals`, `held_s()` and logs
a billable line on release/expiry. Two changes: set `label` to the verified pubkey, and emit
the same line on window grant/extend. That is the metering record; no new billing system.

---

## 4. The x402 payment endpoint

We use the **x402 pattern** (HTTP `402` → client attaches payment proof → server verifies)
over **Stellar rails**, not the EVM/facilitator scheme. The artist self-custodies and submits
their own payment; the proxy verifies it *landed*.

### 4.1 The challenge — `402 Payment Required`
Returned by acquire/renew/tool calls when `HVYM_REQUIRE_PAYMENT` and not `is_paid`. Body:

```json
{
  "x402": {
    "asset":     "USDC",
    "issuer":    "<HVYM_USDC_ISSUER G...>",   // omit/"native" for XLM
    "amount":    "0.75",                       // window_s/60 * price_per_min, quantized
    "pay_to":    "<HVYM_PAYEE_ADDRESS G...>",
    "network":   "public",                     // or "testnet"
    "window_s":  900,
    "memo":      "<price_id>",                 // MEMO_TEXT the client must set on the tx
    "price_id":  "<opaque, binds this quote>",
    "horizon":   "https://horizon.stellar.org"
  }
}
```

`price_id`/`memo` bind a payment to this quote+identity so a tx cannot be replayed for a
different identity or a stale price. Keep issued `price_id`s with their `(pubkey, amount,
window_s, expiry)` for a few minutes.

### 4.2 The settlement — `POST /warm/pay` (NEW route)
Signed identity required (§2). The client, having submitted its Stellar payment to Horizon
itself, retries with proof:

```
POST /warm/pay
X-Ink-Pubkey: G...
X-Ink-Auth:   ...
X-Payment:    <stellar transaction hash>        // or a JSON body {"tx":"...","price_id":"..."}
```

**Verify on Horizon** (read-only; `stellar_sdk.Server`):
1. Fetch the transaction by hash; require `successful == true` on the configured network.
2. It contains a **payment operation** to `HVYM_PAYEE_ADDRESS`, asset == configured
   (USDC+issuer, or native XLM), `amount >= quoted amount`.
3. `memo` (MEMO_TEXT) == the `price_id` from the challenge (binds identity+quote).
4. The payment's **source/sender account == the verified pubkey** (the identity paid for
   itself), OR accept any sender if the memo already binds identity — pick one and document
   it; source==identity is stronger, so prefer it.
5. `tx_hash not in consumed_tx` (idempotency). Add it atomically on success.

On success: apply the credit (§3, settle-on-grant) and return the current warm view
(`WarmPool.status()` shape) so the client sees `state`/`expires_at` immediately.

Failure modes → precise status:
- unverifiable / not found / wrong network → `402` again (with a fresh challenge).
- wrong payee/asset/amount/memo → `402`, `detail` naming the mismatch.
- already consumed → treat as success-idempotent (extend once), never double-credit.

### 4.3 Price quote — `GET /warm/price` (optional, for UI)
Open, read-only. Returns `{asset, amount_per_min, window_s, pay_to, network}` so the client can
show "~$0.05/min · ~48 min in wallet" without triggering a 402. Nice-to-have, not a blocker.

---

## 5. Config (all via env, like the rest of the proxy)

| Env | Default | Meaning |
|---|---|---|
| `HVYM_REQUIRE_SIGNED_IDENTITY` | `false` | enforce §2 (else verify-and-log) |
| `HVYM_REQUIRE_PAYMENT` | `false` | enforce §3–§4 (the meter) |
| `HVYM_SIGN_WINDOW_S` | `120` | signature freshness + nonce-replay TTL |
| `HVYM_AI_WINDOW_S` | `900` | paid window length (~15 min) |
| `HVYM_AI_PRICE_PER_MIN` | `0.05` | USD/min; amount = window_s/60 × this |
| `HVYM_SETTLE_ASSET` | `USDC` | `USDC` or `XLM` |
| `HVYM_USDC_ISSUER` | — | Stellar issuer G... for USDC (required if USDC) |
| `HVYM_PAYEE_ADDRESS` | — | HEAVYMETA receiving G... (required) |
| `HVYM_STELLAR_NETWORK` | `public` | `public` or `testnet` |
| `HVYM_HORIZON_URL` | mainnet Horizon | Horizon base for verification |
| `HVYM_BILLING_DB` | `./billing.sqlite` | durable `paid_through` + `consumed_tx` |

Settle in **USDC** (USD-stable, no peg mechanism); XLM would need spot-pricing at pay time
(`AI_BILLING_INTEGRATION.md` §8.1). Keep XLM as a config option but default USDC.

---

## 6. Build checklist / phases

Mirrors `AI_BILLING_INTEGRATION.md` §7 (proxy-side rows):

| Phase | Work | Done when |
|---|---|---|
| **0 — validate rate** | Reconcile `USD_PER_SECOND` ($1.12/hr) against `/billing/endpoints` after a real day (WARMING.md §Open). | pricing confirmed before the meter turns on |
| **2 — identity + window** ✅ | §2 signature verify (log-only first) + §3 `paid_through`/`consumed_tx` store, `Lease.label` = verified pubkey. | verified identity threaded through `/warm` + `/tools/*`; flags default off |
| **3 — x402 endpoint** ✅ | §4: `402` challenge on unpaid acquire/renew/tool; `POST /warm/pay` Horizon verify + settle-on-grant; `GET /warm/price`. | mocked coverage green **and** `scripts/x402_testnet_rehearsal.py` green against real testnet Horizon (25/25) |
| **cutover** | Flip `HVYM_REQUIRE_SIGNED_IDENTITY` then `HVYM_REQUIRE_PAYMENT`. Keep `X-API-Key` as coarse gate. | production is pay-to-play |

### What is actually built, and where

| §  | Lives in | Notes |
|---|---|---|
| 2 | `hvym_img_tools/core/identity.py` | strkey decode, ed25519 verify, request binding, freshness, nonce replay |
| 3 | `hvym_img_tools/billing.py` — `WindowStore` | SQLite; `paid`/`consumed_tx`/`quotes`. Settle-on-grant via `WarmPool(on_warm=...)` |
| 4 | `hvym_img_tools/billing.py` — `Horizon`, `Billing` | read-only Horizon; `PaymentRequired` → the 402 body |
| routes | `hvym_img_tools/proxy.py` | gates on `/warm` + `/tools/*`; new `POST /warm/pay`, `GET /warm/price` |
| 5 | `scripts/install_proxy.sh`, `docker/Dockerfile.proxy` | env written commented-off; `/data` volume for the store |
| 7 | `tests/test_billing.py` | 66 tests, Horizon mocked, no chain and no network |

**Neither `stellar_sdk` nor a facilitator is a dependency.** The strkey codec is
~40 lines and Horizon is three read-only GETs over the `httpx` the proxy already
had, which keeps the always-on image small (its whole design constraint). The
codec is cross-checked against the real `stellar_sdk` in the test suite, where
the dependency is free.

### Before the cutover

1. **Phase 0 first.** The meter should not turn on against an unreconciled rate.
2. **Re-run the testnet rehearsal** — `uv run python scripts/x402_testnet_rehearsal.py`.
   It closes the one row of §7 a mock cannot: the shape of what Horizon really
   returns for a `credit_alphanum4` payment. It also proves both security
   bindings against genuine on-chain payments that must still be refused — a
   lookalike `USDC` issuer, and a stranger paying someone else's quote.
3. **Establish the payee's USDC trustline *before* announcing a price.** On
   Stellar an account cannot receive an issued asset without one, so until it
   exists every payment fails with `op_no_trust` at the payer's end and the
   artist sees a failure that looks like our bug. This is not hypothetical — it
   happened during the testnet run, to a send made before the trustline existed.
4. **Then flip, in order**, watching the logs between: signed identity, then
   payment.

(Phase 1 and 4 are the client's; already shipped / pending in Inkternity.)

---

## 7. Tests

- **Signature:** valid passes; tampered payload, wrong pubkey, expired `t`, replayed `n`,
  method/path/tool/lid mismatch each `401`. Strkey decode round-trips.
- **Window:** `is_paid` boundary; grant credit; **settle-on-grant** (credit not consumed
  until `warm`); restart reloads `paid_through`.
- **Payment (Horizon mocked):** correct tx extends; wrong payee/asset/amount/memo/network
  rejected; consumed-tx idempotent (no double credit); sender==identity enforced.
- **402 shape:** unpaid acquire/renew/tool returns the §4.1 body.
- **Compat:** with both flags `false`, today's client (X-API-Key only) still works unchanged.
- **Integration:** Stellar **testnet** end-to-end — 402 → pay → `/warm/pay` → warm.

---

## 8. Explicitly out of scope (do not build)

- **Free tier / trial minutes.** There is none (§0). No portal identity, nothing to Sybil.
- **Server-held balances, a ledger, refunds beyond settle-on-grant.** State is one timestamp
  per pubkey + a consumed-tx set.
- **Fiat, merchant-of-record, on-ramp.** Acquiring crypto is the user's problem.
- **A portal handshake / `is_trusted` registry for billing.** That is C2PA publishing's
  concern; the paid path self-authenticates and needs no trust anchor.

---

## 9. Open questions

1. **USDC vs XLM** — *settled as built:* USDC by default, XLM behind
   `HVYM_SETTLE_ASSET=XLM`. XLM still has no spot pricing, so that path quotes in
   XLM units, not dollars — do not turn it on without solving that first.
2. **`paid_through` backend** — *settled as built:* SQLite. `WindowStore` is the
   only thing that touches it, so Redis remains a one-class swap if it is ever
   wanted. Note the one coupling: the nonce replay cache is **in-process**, which
   is correct for one instance and wrong the moment this runs behind a load
   balancer — a nonce seen by one instance is unseen by its sibling.
3. **Sender binding** — *settled as built:* source == identity, enforced, with
   `HVYM_BIND_PAYER_IDENTITY=false` as the escape hatch if the client's submit
   path turns out to pay from a different account. **Still worth confirming
   against the real client before cutover** — this is the one decision here that
   a wrong guess turns into "nobody can pay".
4. **Window length** — 900 s, `HVYM_AI_WINDOW_S`. Unchanged.
5. **Refund/credit for a window cut short by our throttling** — **still open.**
   Settle-on-grant covers "never warmed"; a window that goes cold *mid-way*
   because we throttled is not covered and would need a second credit path.
