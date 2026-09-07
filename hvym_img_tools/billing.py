"""x402 over Stellar rails — the paid-window meter (docs/X402_BILLING.md).

**What this is not.** Not a ledger, not an accounts table, not a wallet. We never
custody funds and never hold a balance. The artist pays peer-to-peer from their
own Stellar wallet to ours; this module's whole job is to check that a payment
*landed on-chain* and, if it did, to extend one timestamp. Total durable state:

    paid_through: pubkey -> unix_ts        # when this identity's window ends
    paid_credit:  pubkey -> seconds        # bought, not yet started (see below)
    consumed_tx:  {tx_hash}                # so one payment cannot pay twice

That smallness is a consequence of one product decision, not a shortcut: **the
inference endpoint is always payment-required, and there is no free tier**
(X402_BILLING.md §0). With no free tier there is nothing to Sybil, so there is
no need for accounts, and identity collapses into "a wallet that can pay" —
which `core.identity` already verifies per request.

**Settle on grant, not on payment.** The clock does not start when money
arrives; it starts the first time a worker actually reaches `warm` for that
identity. A window consumed by a cold start the artist waited through and never
got is a window we charged for and did not deliver (§3, "no warm, no charge").
So a payment buys a *credit*, and `WarmPool` cashes it in via `on_warm`.

**Why no `stellar_sdk` here either.** Verification is three read-only Horizon
GETs and some string comparison. `httpx` is already in the proxy image;
`stellar_sdk` would not be. See `core.identity` for the same reasoning about the
signature side.

**Two flags, so this ships before it bites** (§2 "Phased rollout"):
`HVYM_REQUIRE_SIGNED_IDENTITY` then `HVYM_REQUIRE_PAYMENT`. With both off — the
default — the proxy behaves exactly as it does today and merely logs what it
would have done. Turn them on in that order, once the logs show real clients
signing.
"""
from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_UP
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

#: Horizon per network, when HVYM_HORIZON_URL is not set explicitly.
HORIZON_URLS = {
    "public": "https://horizon.stellar.org",
    "testnet": "https://horizon-testnet.stellar.org",
}

#: What Horizon reports at its root. Checked once, because "the transaction is
#: valid" is meaningless without "...on the network we are being paid on": a
#: testnet payment is free to make and would otherwise buy real GPU time.
NETWORK_PASSPHRASES = {
    "public": "Public Global Stellar Network ; September 2015",
    "testnet": "Test SDF Network ; September 2015",
}

#: Stellar amounts carry 7 decimal places. Quote at the same precision so the
#: string we ask for is a string the wallet can pay exactly.
STROOP = Decimal("0.0000001")

#: How long a quote stays payable. Long enough to open a wallet and confirm,
#: short enough that a stale price cannot be paid at.
QUOTE_TTL_S = 600.0

#: MEMO_TEXT is 28 bytes. 8 random bytes of hex is 16 characters -- unguessable
#: enough to bind a quote, small enough to leave room.
_PRICE_ID_BYTES = 8


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


class PaymentRequired(Exception):
    """No live paid window. Carries the §4.1 challenge the client must satisfy."""

    def __init__(self, challenge: dict, detail: str = "payment required") -> None:
        super().__init__(detail)
        self.challenge = challenge
        self.detail = detail


class PaymentInvalid(Exception):
    """A submitted payment we would not accept, and which check refused it.

    The message names the mismatch (payee, asset, amount, memo, network, sender)
    because a client that paid the wrong thing can only fix it if we say what
    was wrong.
    """


# ------------------------------------------------------------------- config
@dataclass(frozen=True, slots=True)
class BillingConfig:
    """Every knob from docs/X402_BILLING.md §5. Environment only, like the rest
    of the proxy — a serverless-adjacent service is configured by env, not files."""

    require_identity: bool = False
    require_payment: bool = False
    sign_window_s: float = 120.0
    window_s: float = 900.0
    price_per_min: Decimal = Decimal("0.05")
    asset: str = "USDC"
    usdc_issuer: str = ""
    payee: str = ""
    network: str = "public"
    horizon_url: str = HORIZON_URLS["public"]
    db_path: Path = Path("./billing.sqlite")
    #: Require the payment's sender to *be* the identity being credited (§9 Q3).
    #: The stronger of the two bindings, and the default: with it off, anyone can
    #: fund anyone's window as long as they know the memo.
    bind_payer_identity: bool = True

    @classmethod
    def from_env(cls) -> "BillingConfig":
        network = os.environ.get("HVYM_STELLAR_NETWORK", "public").strip().lower()
        if network not in NETWORK_PASSPHRASES:
            log.warning("HVYM_STELLAR_NETWORK=%r is not public/testnet; using public", network)
            network = "public"
        return cls(
            require_identity=_env_bool("HVYM_REQUIRE_SIGNED_IDENTITY", False),
            require_payment=_env_bool("HVYM_REQUIRE_PAYMENT", False),
            sign_window_s=float(os.environ.get("HVYM_SIGN_WINDOW_S", "120")),
            window_s=float(os.environ.get("HVYM_AI_WINDOW_S", "900")),
            price_per_min=Decimal(os.environ.get("HVYM_AI_PRICE_PER_MIN", "0.05")),
            asset=os.environ.get("HVYM_SETTLE_ASSET", "USDC").strip().upper() or "USDC",
            usdc_issuer=os.environ.get("HVYM_USDC_ISSUER", "").strip(),
            payee=os.environ.get("HVYM_PAYEE_ADDRESS", "").strip(),
            network=network,
            horizon_url=(
                os.environ.get("HVYM_HORIZON_URL", "").strip() or HORIZON_URLS[network]
            ).rstrip("/"),
            db_path=Path(os.environ.get("HVYM_BILLING_DB", "./billing.sqlite")).expanduser(),
            bind_payer_identity=_env_bool("HVYM_BIND_PAYER_IDENTITY", True),
        )

    @property
    def native(self) -> bool:
        return self.asset == "XLM"

    @property
    def amount(self) -> str:
        """What one window costs, as the exact string the client must pay.

        Rounded **up** to the stroop: rounding a price down would let a window be
        bought for fractionally less than it costs, forever, at scale.
        """
        raw = Decimal(self.window_s) / Decimal(60) * self.price_per_min
        return format(raw.quantize(STROOP, rounding=ROUND_UP).normalize(), "f")

    def misconfiguration(self) -> str | None:
        """Why payment cannot be enforced yet, or None. Checked at startup so a
        half-configured meter fails loudly instead of 500-ing the first artist."""
        if not self.payee:
            return "HVYM_PAYEE_ADDRESS is not set"
        if not self.native and not self.usdc_issuer:
            return f"HVYM_SETTLE_ASSET={self.asset} needs HVYM_USDC_ISSUER"
        if self.price_per_min <= 0:
            return "HVYM_AI_PRICE_PER_MIN must be positive"
        if self.window_s <= 0:
            return "HVYM_AI_WINDOW_S must be positive"
        return None


@dataclass(frozen=True, slots=True)
class Quote:
    """One issued price, bound to one identity. `price_id` is the MEMO_TEXT the
    payment must carry, which is what stops a transaction paid for one identity
    (or at a stale price) from being redeemed for another."""

    price_id: str
    pubkey: str
    amount: str
    window_s: float
    expires_at: float


# -------------------------------------------------------------------- store
class WindowStore:
    """`paid_through` + `paid_credit` + `consumed_tx`, on SQLite.

    SQLite because the state is three small tables on one always-on box and
    zero-ops beats fast here; the interface is narrow enough that swapping in
    Redis later touches only this class (§9 Q2).

    **It must be durable.** The artist paid. A proxy restart that forgot a window
    is a refund request, so `HVYM_BILLING_DB` has to point at something that
    outlives the container — see `scripts/install_proxy.sh`, which mounts a
    volume for exactly this.
    """

    def __init__(self, path: Path | str = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS paid ("
                " pubkey TEXT PRIMARY KEY,"
                " paid_through REAL NOT NULL DEFAULT 0,"
                " credit_s REAL NOT NULL DEFAULT 0)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS consumed_tx ("
                " tx_hash TEXT PRIMARY KEY, pubkey TEXT NOT NULL, at REAL NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS quotes ("
                " price_id TEXT PRIMARY KEY, pubkey TEXT NOT NULL, amount TEXT NOT NULL,"
                " window_s REAL NOT NULL, expires_at REAL NOT NULL)"
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- windows ----------------------------------------------------------
    def row(self, pubkey: str) -> tuple[float, float]:
        """`(paid_through, credit_s)` for an identity; zeros if unknown."""
        with self._lock:
            found = self._db.execute(
                "SELECT paid_through, credit_s FROM paid WHERE pubkey=?", (pubkey,)
            ).fetchone()
        return (float(found["paid_through"]), float(found["credit_s"])) if found else (0.0, 0.0)

    def entitled(self, pubkey: str, now: float) -> bool:
        """Whether this identity may spend GPU time right now.

        Note the `or`: an outstanding credit counts. Settle-on-grant means a
        just-paid artist has no `paid_through` yet — their window starts when the
        worker warms — so gating on `paid_through` alone would 402 the very
        `POST /warm` that is supposed to start their clock.
        """
        paid_through, credit_s = self.row(pubkey)
        return now < paid_through or credit_s > 0

    def credit_for_tx(self, tx_hash: str, pubkey: str, window_s: float, now: float) -> bool:
        """Consume a transaction and grant its window, atomically.

        Returns False if the hash was already spent — which is *success* for the
        caller (a client retrying a dropped response must not be told its payment
        failed) but must not credit a second window. Both writes share one
        transaction so a crash between them cannot lose a payment we already
        marked as spent.
        """
        with self._lock, self._db:
            try:
                self._db.execute(
                    "INSERT INTO consumed_tx (tx_hash, pubkey, at) VALUES (?,?,?)",
                    (tx_hash, pubkey, now),
                )
            except sqlite3.IntegrityError:
                return False
            self._db.execute(
                "INSERT INTO paid (pubkey, paid_through, credit_s) VALUES (?,0,?)"
                " ON CONFLICT(pubkey) DO UPDATE SET credit_s = credit_s + excluded.credit_s",
                (pubkey, window_s),
            )
        return True

    def settle(self, pubkey: str, now: float) -> float | None:
        """Start the clock on any outstanding credit. Returns the new
        `paid_through`, or None when there was nothing to settle.

        Extends from `max(now, paid_through)` rather than from `now`: an artist
        who tops up mid-window bought more time, and must not have the remainder
        of the window they already paid for silently truncated.
        """
        with self._lock, self._db:
            found = self._db.execute(
                "SELECT paid_through, credit_s FROM paid WHERE pubkey=?", (pubkey,)
            ).fetchone()
            if not found or float(found["credit_s"]) <= 0:
                return None
            new_through = max(now, float(found["paid_through"])) + float(found["credit_s"])
            self._db.execute(
                "UPDATE paid SET paid_through=?, credit_s=0 WHERE pubkey=?", (new_through, pubkey)
            )
        return new_through

    # -- quotes -----------------------------------------------------------
    def issue_quote(self, pubkey: str, amount: str, window_s: float, now: float) -> Quote:
        """Return a live quote for this identity, minting one only if needed.

        Reuse matters: a client that gets 402'd, pays, and is 402'd again before
        settling must see the *same* memo, or the payment it is about to make
        would be bound to a quote we had already forgotten.
        """
        with self._lock, self._db:
            self._db.execute("DELETE FROM quotes WHERE expires_at <= ?", (now,))
            found = self._db.execute(
                "SELECT * FROM quotes WHERE pubkey=? AND amount=? AND window_s=?"
                " ORDER BY expires_at DESC LIMIT 1",
                (pubkey, amount, window_s),
            ).fetchone()
            if found:
                return Quote(
                    found["price_id"], found["pubkey"], found["amount"],
                    float(found["window_s"]), float(found["expires_at"]),
                )
            quote = Quote(
                price_id=secrets.token_hex(_PRICE_ID_BYTES),
                pubkey=pubkey,
                amount=amount,
                window_s=window_s,
                expires_at=now + QUOTE_TTL_S,
            )
            self._db.execute(
                "INSERT INTO quotes (price_id, pubkey, amount, window_s, expires_at)"
                " VALUES (?,?,?,?,?)",
                (quote.price_id, quote.pubkey, quote.amount, quote.window_s, quote.expires_at),
            )
        return quote

    def quote(self, price_id: str, now: float) -> Quote | None:
        with self._lock:
            found = self._db.execute(
                "SELECT * FROM quotes WHERE price_id=?", (price_id,)
            ).fetchone()
        if not found or float(found["expires_at"]) <= now:
            return None
        return Quote(
            found["price_id"], found["pubkey"], found["amount"],
            float(found["window_s"]), float(found["expires_at"]),
        )


# ------------------------------------------------------------------ horizon
class Horizon:
    """Read-only Horizon checks. Never signs, never submits, holds no key.

    The client submits its own payment; all we do is look it up. That is the
    whole reason this is not a payment processor: there is no path from a
    compromised proxy to someone else's funds.
    """

    def __init__(self, config: BillingConfig, *, client_factory=httpx.AsyncClient,
                 timeout: float = 15.0) -> None:
        self._cfg = config
        self._client_factory = client_factory
        self._timeout = timeout
        self._passphrase: str | None = None

    async def _get(self, path: str) -> dict:
        url = f"{self._cfg.horizon_url}{path}"
        async with self._client_factory(timeout=self._timeout) as client:
            resp = await client.get(url, headers={"Accept": "application/json"})
        if resp.status_code == 404:
            raise PaymentInvalid("transaction not found on this network")
        if resp.status_code >= 400:
            raise PaymentInvalid(f"horizon returned {resp.status_code}")
        try:
            return resp.json() or {}
        except ValueError as exc:
            raise PaymentInvalid("horizon returned a non-JSON body") from exc

    async def _check_network(self) -> None:
        """Confirm the configured Horizon really serves the configured network.

        Cached after the first success. Pointing `HVYM_HORIZON_URL` at testnet
        while charging in public-network dollars is the one misconfiguration that
        makes the whole meter free, so it is checked rather than assumed.
        """
        if self._passphrase is not None:
            return
        root = await self._get("/")
        seen = str(root.get("network_passphrase") or "")
        expected = NETWORK_PASSPHRASES[self._cfg.network]
        if seen != expected:
            raise PaymentInvalid(
                f"horizon serves {seen or 'an unknown network'}, not {self._cfg.network}"
            )
        self._passphrase = seen

    def _asset_matches(self, op: dict) -> bool:
        if self._cfg.native:
            return op.get("asset_type") == "native"
        return (
            op.get("asset_type") in {"credit_alphanum4", "credit_alphanum12"}
            and op.get("asset_code") == self._cfg.asset
            and op.get("asset_issuer") == self._cfg.usdc_issuer
        )

    async def verify_payment(self, tx_hash: str, quote: Quote) -> str:
        """Every check in §4.2. Returns the amount paid; raises `PaymentInvalid`.

        Deliberately strict about *which* operation counts: a transaction may
        carry many, and only one that pays the configured asset to the configured
        payee for at least the quoted amount settles this quote.
        """
        if not tx_hash or len(tx_hash) != 64 or not all(c in "0123456789abcdefABCDEF" for c in tx_hash):
            raise PaymentInvalid("payment proof is not a 64-hex transaction hash")
        await self._check_network()

        tx = await self._get(f"/transactions/{tx_hash.lower()}")
        if not tx.get("successful", False):
            raise PaymentInvalid("transaction did not succeed on-chain")
        if tx.get("memo_type") != "text" or str(tx.get("memo") or "") != quote.price_id:
            raise PaymentInvalid("transaction memo does not carry this quote's price_id")

        ops = await self._get(f"/transactions/{tx_hash.lower()}/operations?limit=200")
        records = ((ops.get("_embedded") or {}).get("records")) or []
        want = Decimal(quote.amount)
        wrong_reason = "transaction contains no payment to us"
        for op in records:
            if op.get("type") != "payment":
                continue
            if op.get("to") != self._cfg.payee:
                wrong_reason = "payment was sent to a different account"
                continue
            if not self._asset_matches(op):
                wrong_reason = f"payment was not in {self._cfg.asset}"
                continue
            try:
                paid = Decimal(str(op.get("amount") or "0"))
            except (ValueError, ArithmeticError):
                continue
            if paid < want:
                wrong_reason = f"payment of {paid} is below the quoted {quote.amount}"
                continue
            if self._cfg.bind_payer_identity and op.get("from") != quote.pubkey:
                wrong_reason = "payment was not sent by the identity being credited"
                continue
            return format(paid.normalize(), "f")
        raise PaymentInvalid(wrong_reason)


# ------------------------------------------------------------------- facade
class Billing:
    """What the proxy talks to. Ties the config, the store and Horizon together
    and owns the two rollout flags."""

    def __init__(
        self,
        config: BillingConfig | None = None,
        *,
        store: WindowStore | None = None,
        horizon: Horizon | None = None,
        clock=time.time,
    ) -> None:
        self.config = config or BillingConfig.from_env()
        self._store = store
        self.horizon = horizon or Horizon(self.config)
        self._clock = clock

    # -- flags ------------------------------------------------------------
    @property
    def require_identity(self) -> bool:
        return self.config.require_identity

    @property
    def require_payment(self) -> bool:
        return self.config.require_payment

    @property
    def store(self) -> WindowStore:
        """Opened on first use, never at construction.

        Every `create_app()` builds a `Billing` -- including the ones in tests and
        in the Dockerfile's import check -- and a deployment that has not turned
        the meter on should not find a stray `billing.sqlite` in its working
        directory as a result. Nothing here touches the disk until something is
        actually charged, quoted, or settled.
        """
        if self._store is None:
            self._store = WindowStore(self.config.db_path)
        return self._store

    @property
    def _has_state(self) -> bool:
        """Whether any window state could exist -- open, or on disk from before a
        restart. Keeps `on_warm` (which fires every few seconds while a lease is
        held) from creating the database just by looking."""
        if self._store is not None:
            return True
        path = str(self.config.db_path)
        return path != ":memory:" and Path(path).exists()

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    # -- gating -----------------------------------------------------------
    def entitled(self, pubkey: str | None) -> bool:
        if not self.require_payment:
            return True
        if not pubkey:
            return False
        return self.store.entitled(pubkey, self._clock())

    def challenge(self, pubkey: str) -> dict:
        """The §4.1 `x402` object: what to pay, to whom, with which memo."""
        cfg = self.config
        quote = self.store.issue_quote(pubkey, cfg.amount, cfg.window_s, self._clock())
        body = {
            "asset": cfg.asset,
            "amount": quote.amount,
            "pay_to": cfg.payee,
            "network": cfg.network,
            "window_s": cfg.window_s,
            "memo": quote.price_id,
            "price_id": quote.price_id,
            "horizon": cfg.horizon_url,
        }
        if not cfg.native:
            body["issuer"] = cfg.usdc_issuer
        return body

    def require(self, pubkey: str | None) -> None:
        """Raise `PaymentRequired` unless this identity has a live window."""
        if self.entitled(pubkey):
            return
        if not pubkey:
            # Enforcing payment without an identity to charge is not a 402 the
            # client can act on -- there is no wallet to quote a price to.
            raise PaymentRequired({}, "a signed wallet identity is required to pay")
        raise PaymentRequired(self.challenge(pubkey), "no live paid window for this identity")

    def price_view(self) -> dict:
        """§4.3 — open, read-only, spends nothing and issues no quote, so a UI
        can show "~$0.05/min" without tripping the meter."""
        cfg = self.config
        view = {
            "asset": cfg.asset,
            "amount": cfg.amount,
            "amount_per_min": format(cfg.price_per_min.normalize(), "f"),
            "window_s": cfg.window_s,
            "pay_to": cfg.payee or None,
            "network": cfg.network,
            "enforced": cfg.require_payment,
        }
        if not cfg.native:
            view["issuer"] = cfg.usdc_issuer or None
        return view

    # -- settlement -------------------------------------------------------
    async def settle_payment(self, pubkey: str, tx_hash: str, price_id: str = "") -> dict:
        """Verify a submitted payment and grant its window as a credit.

        Idempotent by transaction hash: a client retrying after a dropped
        response gets the same success, and is credited once.
        """
        now = self._clock()
        quote = self.store.quote(price_id, now) if price_id else None
        if quote is None:
            # No price_id, or a stale one: fall back to this identity's live
            # quote. That is not a loophole -- the memo still has to match it.
            quote = self.store.issue_quote(pubkey, self.config.amount, self.config.window_s, now)
        if quote.pubkey != pubkey:
            raise PaymentInvalid("this quote was issued to a different identity")

        amount = await self.horizon.verify_payment(tx_hash, quote)
        granted = self.store.credit_for_tx(tx_hash, pubkey, quote.window_s, now)
        if granted:
            # The metering record, matching warm.py's release/expiry line.
            log.info(
                "warm window granted pubkey=%s window=%.0fs amount=%s %s tx=%s",
                pubkey[:8], quote.window_s, amount, self.config.asset, tx_hash[:12],
            )
        else:
            log.info("payment already settled, not re-crediting tx=%s", tx_hash[:12])
        paid_through, credit_s = self.store.row(pubkey)
        return {
            "settled": True,
            "credited": granted,
            "amount": amount,
            "asset": self.config.asset,
            "window_s": quote.window_s,
            "paid_through": paid_through or None,
            "pending_credit_s": credit_s,
        }

    def on_warm(self, labels) -> None:
        """`WarmPool` telling us a worker actually reached `warm`.

        This is the settle-on-grant transition (§3): until it fires, a paid
        window has been bought but not started, so an artist who paid and then
        sat through a cold start that never arrived keeps their time.
        """
        if not self._has_state:
            return
        now = self._clock()
        for label in labels:
            if not label:
                continue
            settled = self.store.settle(label, now)
            if settled is not None:
                log.info(
                    "warm window started pubkey=%s paid_through=+%.0fs",
                    label[:8], settled - now,
                )
