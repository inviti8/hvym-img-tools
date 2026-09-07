"""x402 billing — signature verification, the paid window, and settlement.

No network and no chain: Horizon is a fake that answers from a dict, so every
on-chain rule (payee, asset, amount, memo, sender, network) can be violated one
at a time and the refusal named.

The signatures here are built the way the *client* builds them — the payload
string is assembled by hand in `sign()`, character for character, rather than
via `json.dumps`. That is the point: `../infinipaint/src/AI/RequestSigner.cpp`
hand-builds those bytes precisely so the two sides never have to agree about a
JSON serializer, and a test that used our own serializer on both sides would
verify nothing about that agreement.
"""
from __future__ import annotations

import asyncio
import base64
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from nacl.signing import SigningKey

from hvym_img_tools import billing as billing_mod
from hvym_img_tools import proxy as proxy_mod
from hvym_img_tools.billing import (
    Billing,
    BillingConfig,
    Horizon,
    PaymentInvalid,
    WindowStore,
)
from hvym_img_tools.core import identity as ident_mod
from hvym_img_tools.core.identity import (
    IdentityError,
    IdentityVerifier,
    encode_ed25519_public_key,
)

USDC_ISSUER = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
PAYEE = "GDQNY3PBOJOKYZSRMK2S7LHHGWZIUISD4QORETLMXEWXBI7KFZZMKTL3"
TX = "a" * 64


# --------------------------------------------------------------- the client
class Wallet:
    """Stands in for Inkternity's DevKeys identity."""

    def __init__(self) -> None:
        self.sk = SigningKey.generate()
        self.pubkey = encode_ed25519_public_key(bytes(self.sk.verify_key))

    def sign(self, method, path, tool, lease_id="", *, t=None, nonce="0" * 32,
             payload=None) -> dict:
        """The two headers, built exactly as RequestSigner.cpp builds them."""
        if payload is None:
            t = int(time.time()) if t is None else t
            payload = (
                '{"i":"ai-request"'
                f',"lid":"{lease_id}"'
                f',"m":"{method}"'
                f',"n":"{nonce}"'
                f',"p":"{path}"'
                f',"t":{t}'
                f',"tool":"{tool}"}}'
            )
        raw = payload.encode()
        sig = self.sk.sign(raw).signature
        return {
            ident_mod.PUBKEY_HEADER: self.pubkey,
            ident_mod.AUTH_HEADER: f"{_b64u(sig)}.{_b64u(raw)}",
        }


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


@pytest.fixture
def wallet():
    return Wallet()


# ------------------------------------------------------------------ strkey
def test_strkey_matches_the_real_stellar_sdk():
    """We decode Stellar addresses by hand to keep `stellar_sdk` out of the proxy
    image. That shortcut is only safe while it agrees with the real thing."""
    sdk = pytest.importorskip("stellar_sdk")
    for _ in range(10):
        kp = sdk.Keypair.random()
        assert ident_mod.decode_ed25519_public_key(kp.public_key) == kp.raw_public_key()
        assert encode_ed25519_public_key(kp.raw_public_key()) == kp.public_key


def test_a_typoed_address_is_rejected_not_reinterpreted(wallet):
    """A mistyped address that happens to be 56 characters must not decode into a
    *different* identity -- that would be a billing window nobody can ever pay."""
    swapped = wallet.pubkey[:20] + ("B" if wallet.pubkey[20] != "B" else "C") + wallet.pubkey[21:]
    with pytest.raises(IdentityError, match="checksum"):
        ident_mod.decode_ed25519_public_key(swapped)


@pytest.mark.parametrize("bad", ["", "G" + "A" * 40, "MDQNY3PBOJOKYZSRMK2S7LHHGWZIUISD4QORETLMXEWXBI7KFZZMKTL3"])
def test_non_pubkeys_are_refused(bad):
    with pytest.raises(IdentityError):
        ident_mod.decode_ed25519_public_key(bad)


# -------------------------------------------------------------- signatures
def _verify(verifier, headers, *, method="POST", path="/warm", tool="reangle", lease_id=""):
    return verifier.verify(
        pubkey_header=headers.get(ident_mod.PUBKEY_HEADER),
        auth_header=headers.get(ident_mod.AUTH_HEADER),
        method=method, path=path, tool=tool, lease_id=lease_id,
    )


def test_a_valid_signature_yields_the_identity(wallet):
    verifier = IdentityVerifier()
    ident = _verify(verifier, wallet.sign("POST", "/warm", "reangle"))
    assert ident.pubkey == wallet.pubkey
    assert ident.tool == "reangle"


def test_a_tampered_payload_does_not_verify(wallet):
    headers = wallet.sign("POST", "/warm", "reangle")
    sig, _, payload = headers[ident_mod.AUTH_HEADER].partition(".")
    tampered = _b64u(base64.urlsafe_b64decode(payload + "==").replace(b"reangle", b"mesh!!!!"))
    headers[ident_mod.AUTH_HEADER] = f"{sig}.{tampered}"
    with pytest.raises(IdentityError, match="does not verify"):
        _verify(verifier=IdentityVerifier(), headers=headers)


def test_another_wallets_signature_is_not_accepted(wallet):
    other = Wallet()
    headers = wallet.sign("POST", "/warm", "reangle")
    headers[ident_mod.PUBKEY_HEADER] = other.pubkey
    with pytest.raises(IdentityError, match="does not verify"):
        _verify(IdentityVerifier(), headers)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"method": "DELETE"}, "method"),
        ({"path": "/tools/mesh"}, "path"),
        ({"tool": "mesh"}, "tool"),
        ({"lease_id": "abc"}, "lease_id"),
    ],
)
def test_a_signature_is_bound_to_the_request_it_arrived_on(wallet, kwargs, expected):
    """The signature covers method, path, tool and lease -- so one lifted from a
    cheap call cannot be replayed against an expensive one."""
    headers = wallet.sign("POST", "/warm", "reangle")
    with pytest.raises(IdentityError, match=expected):
        _verify(IdentityVerifier(), headers, **kwargs)


def test_a_signature_from_last_hour_is_stale(wallet):
    verifier = IdentityVerifier(window_s=120)
    headers = wallet.sign("POST", "/warm", "reangle", t=int(time.time()) - 3600)
    with pytest.raises(IdentityError, match="freshness"):
        _verify(verifier, headers)


def test_a_signature_from_the_future_is_also_stale(wallet):
    verifier = IdentityVerifier(window_s=120)
    headers = wallet.sign("POST", "/warm", "reangle", t=int(time.time()) + 3600)
    with pytest.raises(IdentityError, match="freshness"):
        _verify(verifier, headers)


def test_the_same_nonce_cannot_be_used_twice(wallet):
    verifier = IdentityVerifier()
    headers = wallet.sign("POST", "/warm", "reangle", nonce="beef" * 8)
    _verify(verifier, headers)
    with pytest.raises(IdentityError, match="already been used"):
        _verify(verifier, headers)


def test_a_rejected_request_does_not_burn_the_nonce(wallet):
    """A malformed or mis-bound request must not consume a nonce the legitimate
    client is about to use -- otherwise anyone who can see one header can lock
    that request out by replaying it against the wrong path first."""
    verifier = IdentityVerifier()
    headers = wallet.sign("POST", "/warm", "reangle", nonce="cafe" * 8)
    with pytest.raises(IdentityError):
        _verify(verifier, headers, path="/tools/mesh")
    assert _verify(verifier, headers).pubkey == wallet.pubkey


def test_replayed_nonces_are_forgotten_once_they_could_no_longer_be_used(wallet):
    """The cache only has to outlive the freshness window: a nonce whose
    signature is already too stale to accept costs nothing to forget."""
    now = [1000.0]
    verifier = IdentityVerifier(window_s=60, clock=lambda: now[0])
    headers = wallet.sign("POST", "/warm", "reangle", t=1000, nonce="dead" * 8)
    _verify(verifier, headers)
    now[0] = 1000.0 + 61
    assert not verifier._seen or all(exp <= now[0] for exp in verifier._seen.values())


@pytest.mark.parametrize("auth", ["", "no-dot", "onlyone.", ".onlypayload", "!!!.###"])
def test_a_malformed_envelope_is_refused(wallet, auth):
    headers = {ident_mod.PUBKEY_HEADER: wallet.pubkey, ident_mod.AUTH_HEADER: auth}
    with pytest.raises(IdentityError):
        _verify(IdentityVerifier(), headers)


def test_a_signature_for_a_different_purpose_is_not_an_ai_request(wallet):
    """Same envelope shape as C2PA's WireToken. The intent tag is what stops one
    being spent as the other."""
    payload = '{"i":"c2pa-publish","lid":"","m":"POST","n":"' + "0" * 32 + '","p":"/warm","t":' \
              + str(int(time.time())) + ',"tool":"reangle"}'
    with pytest.raises(IdentityError, match="ai-request"):
        _verify(IdentityVerifier(), wallet.sign("POST", "/warm", "reangle", payload=payload))


# ------------------------------------------------------------- the window
def _config(**kw) -> BillingConfig:
    base = dict(
        require_identity=True, require_payment=True, payee=PAYEE,
        usdc_issuer=USDC_ISSUER, window_s=900.0, network="testnet",
        horizon_url="https://horizon-testnet.stellar.org", db_path=":memory:",
    )
    base.update(kw)
    return BillingConfig(**base)


def test_price_is_window_times_rate():
    assert _config(window_s=900, price_per_min=billing_mod.Decimal("0.05")).amount == "0.75"
    assert _config(window_s=600, price_per_min=billing_mod.Decimal("0.05")).amount == "0.5"


def test_a_fractional_price_rounds_up_not_down():
    """Rounding a price down is a discount we grant forever, at every scale."""
    cfg = _config(window_s=61, price_per_min=billing_mod.Decimal("0.0333333"))
    assert billing_mod.Decimal(cfg.amount) >= billing_mod.Decimal(61) / 60 * cfg.price_per_min


def test_enforcing_payment_without_a_payee_is_a_misconfiguration():
    assert "HVYM_PAYEE_ADDRESS" in (_config(payee="").misconfiguration() or "")
    assert "HVYM_USDC_ISSUER" in (_config(usdc_issuer="").misconfiguration() or "")
    assert _config().misconfiguration() is None


def test_an_unpaid_identity_is_not_entitled(wallet):
    store = WindowStore(":memory:")
    assert not store.entitled(wallet.pubkey, now=1000.0)


def test_a_credit_entitles_before_the_clock_has_started(wallet):
    """Settle-on-grant means a just-paid artist has no `paid_through` yet. If the
    gate looked only at that, it would 402 the very POST /warm that is supposed
    to start their window."""
    store = WindowStore(":memory:")
    store.credit_for_tx(TX, wallet.pubkey, 900.0, now=1000.0)
    assert store.entitled(wallet.pubkey, now=1000.0)
    assert store.row(wallet.pubkey) == (0.0, 900.0)


def test_the_window_starts_when_the_worker_warms_not_when_money_arrives(wallet):
    """docs/X402_BILLING.md §3, "no warm, no charge". An artist who paid and then
    waited out a cold start that never arrived keeps their time."""
    store = WindowStore(":memory:")
    store.credit_for_tx(TX, wallet.pubkey, 900.0, now=1000.0)
    assert store.row(wallet.pubkey)[0] == 0.0            # nothing running yet
    much_later = 1000.0 + 4000                            # they waited, then warmed
    assert store.settle(wallet.pubkey, much_later) == much_later + 900.0
    assert store.entitled(wallet.pubkey, much_later + 899)
    assert not store.entitled(wallet.pubkey, much_later + 901)


def test_settling_twice_does_not_extend_twice(wallet):
    store = WindowStore(":memory:")
    store.credit_for_tx(TX, wallet.pubkey, 900.0, now=1000.0)
    assert store.settle(wallet.pubkey, 1000.0) == 1900.0
    assert store.settle(wallet.pubkey, 1000.0) is None


def test_topping_up_mid_window_extends_it_rather_than_truncating_it(wallet):
    """An artist with 400s left who buys another 900 must end up with 1300, not
    900 -- settling from `now` would silently confiscate the remainder."""
    store = WindowStore(":memory:")
    store.credit_for_tx(TX, wallet.pubkey, 900.0, now=1000.0)
    store.settle(wallet.pubkey, 1000.0)                    # paid_through = 1900
    store.credit_for_tx("b" * 64, wallet.pubkey, 900.0, now=1500.0)
    assert store.settle(wallet.pubkey, 1500.0) == 2800.0


def test_one_transaction_cannot_be_spent_twice(wallet):
    store = WindowStore(":memory:")
    assert store.credit_for_tx(TX, wallet.pubkey, 900.0, now=1000.0) is True
    assert store.credit_for_tx(TX, wallet.pubkey, 900.0, now=1000.0) is False
    assert store.row(wallet.pubkey)[1] == 900.0


def test_a_paid_window_survives_a_restart(tmp_path, wallet):
    """The artist paid. A proxy restart that forgets a window is a refund request."""
    path = tmp_path / "billing.sqlite"
    store = WindowStore(path)
    store.credit_for_tx(TX, wallet.pubkey, 900.0, now=1000.0)
    store.settle(wallet.pubkey, 1000.0)
    store.close()

    reopened = WindowStore(path)
    assert reopened.entitled(wallet.pubkey, now=1500.0)
    assert reopened.credit_for_tx(TX, wallet.pubkey, 900.0, now=1500.0) is False
    reopened.close()


def test_a_quote_is_reused_while_it_is_live(wallet):
    """A client 402'd twice must see the same memo, or the payment it is about to
    make would be bound to a quote we had already replaced."""
    store = WindowStore(":memory:")
    first = store.issue_quote(wallet.pubkey, "0.75", 900.0, now=1000.0)
    again = store.issue_quote(wallet.pubkey, "0.75", 900.0, now=1100.0)
    assert first.price_id == again.price_id
    assert store.quote(first.price_id, 1100.0) is not None
    assert store.quote(first.price_id, 1000.0 + billing_mod.QUOTE_TTL_S + 1) is None


def test_a_quote_fits_in_a_stellar_text_memo(wallet):
    """MEMO_TEXT is 28 bytes. A price_id that does not fit cannot be paid at all."""
    quote = WindowStore(":memory:").issue_quote(wallet.pubkey, "0.75", 900.0, now=0.0)
    assert len(quote.price_id.encode()) <= 28


def test_two_identities_get_different_quotes(wallet):
    store, other = WindowStore(":memory:"), Wallet()
    a = store.issue_quote(wallet.pubkey, "0.75", 900.0, now=1000.0)
    b = store.issue_quote(other.pubkey, "0.75", 900.0, now=1000.0)
    assert a.price_id != b.price_id


# --------------------------------------------------------------- on-chain
class FakeHorizon:
    """Horizon, from a dict. Every rule can be broken one at a time."""

    def __init__(self, *, passphrase=None, tx=None, ops=None, missing=False):
        self.passphrase = passphrase or billing_mod.NETWORK_PASSPHRASES["testnet"]
        self.missing = missing
        self.tx = tx if tx is not None else {
            "successful": True, "memo_type": "text", "memo": None,   # filled by the test
            "source_account": None,
        }
        self.ops = ops
        self.seen: list[str] = []

    def payment(self, *, to=PAYEE, sender=None, amount="0.75", asset="USDC"):
        op = {"type": "payment", "to": to, "from": sender, "amount": amount}
        if asset == "XLM":
            op["asset_type"] = "native"
        else:
            op.update(asset_type="credit_alphanum4", asset_code=asset, asset_issuer=USDC_ISSUER)
        self.ops = [op]
        return self

    def factory(self):
        horizon = self

        class _Client:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, headers=None):
                horizon.seen.append(url)
                request = httpx.Request("GET", url)
                if url.endswith("/"):
                    return httpx.Response(
                        200, json={"network_passphrase": horizon.passphrase}, request=request
                    )
                if horizon.missing:
                    return httpx.Response(404, json={}, request=request)
                if url.endswith("/operations?limit=200"):
                    return httpx.Response(
                        200, json={"_embedded": {"records": horizon.ops or []}}, request=request
                    )
                return httpx.Response(200, json=horizon.tx, request=request)

        return _Client


async def _settle(fake, wallet, *, config=None, store=None, tx=TX):
    cfg = config or _config()
    store = store or WindowStore(":memory:")
    quote = store.issue_quote(wallet.pubkey, cfg.amount, cfg.window_s, now=1000.0)
    fake.tx.setdefault("memo_type", "text")
    if fake.tx.get("memo") is None:
        fake.tx["memo"] = quote.price_id
    bill = Billing(cfg, store=store, horizon=Horizon(cfg, client_factory=fake.factory()),
                   clock=lambda: 1000.0)
    return await bill.settle_payment(wallet.pubkey, tx, quote.price_id), bill


def test_a_correct_payment_grants_a_window(wallet):
    fake = FakeHorizon().payment(sender=wallet.pubkey)
    result, bill = asyncio.run(_settle(fake, wallet))
    assert result["credited"] is True
    assert bill.entitled(wallet.pubkey)


def test_paying_more_than_quoted_is_fine(wallet):
    fake = FakeHorizon().payment(sender=wallet.pubkey, amount="1.5")
    result, _ = asyncio.run(_settle(fake, wallet))
    assert result["credited"] is True


def test_paying_less_than_quoted_is_refused(wallet):
    fake = FakeHorizon().payment(sender=wallet.pubkey, amount="0.0000001")
    with pytest.raises(PaymentInvalid, match="below the quoted"):
        asyncio.run(_settle(fake, wallet))


def test_paying_the_wrong_account_is_refused(wallet):
    fake = FakeHorizon().payment(to=USDC_ISSUER, sender=wallet.pubkey)
    with pytest.raises(PaymentInvalid, match="different account"):
        asyncio.run(_settle(fake, wallet))


def test_paying_the_wrong_asset_is_refused(wallet):
    fake = FakeHorizon().payment(sender=wallet.pubkey, asset="XLM")
    with pytest.raises(PaymentInvalid, match="not in USDC"):
        asyncio.run(_settle(fake, wallet))


def test_a_payment_from_someone_else_is_refused(wallet):
    """§9 Q3, the stronger binding: the identity being credited must be the one
    that paid. Without it, knowing a memo is enough to top up any wallet."""
    fake = FakeHorizon().payment(sender=Wallet().pubkey)
    with pytest.raises(PaymentInvalid, match="not sent by the identity"):
        asyncio.run(_settle(fake, wallet))


def test_a_third_party_may_pay_when_that_binding_is_relaxed(wallet):
    fake = FakeHorizon().payment(sender=Wallet().pubkey)
    result, _ = asyncio.run(_settle(fake, wallet, config=_config(bind_payer_identity=False)))
    assert result["credited"] is True


def test_the_wrong_memo_is_refused(wallet):
    """The memo is what binds a transaction to one quote and one identity."""
    fake = FakeHorizon(tx={"successful": True, "memo_type": "text", "memo": "somebody-elses"})
    fake.payment(sender=wallet.pubkey)
    with pytest.raises(PaymentInvalid, match="memo"):
        asyncio.run(_settle(fake, wallet))


def test_a_failed_transaction_is_refused(wallet):
    fake = FakeHorizon(tx={"successful": False, "memo_type": "text", "memo": None})
    fake.payment(sender=wallet.pubkey)
    with pytest.raises(PaymentInvalid, match="did not succeed"):
        asyncio.run(_settle(fake, wallet))


def test_an_unknown_transaction_is_refused(wallet):
    fake = FakeHorizon(missing=True).payment(sender=wallet.pubkey)
    with pytest.raises(PaymentInvalid, match="not found"):
        asyncio.run(_settle(fake, wallet))


def test_a_testnet_horizon_cannot_settle_a_mainnet_price(wallet):
    """Testnet lumens are free. Pointing Horizon at the wrong network is the one
    misconfiguration that makes the entire meter free, so it is checked."""
    fake = FakeHorizon(passphrase=billing_mod.NETWORK_PASSPHRASES["testnet"])
    fake.payment(sender=wallet.pubkey)
    with pytest.raises(PaymentInvalid, match="not public"):
        asyncio.run(_settle(fake, wallet, config=_config(network="public")))


def test_a_transaction_hash_that_is_not_one_is_refused(wallet):
    fake = FakeHorizon().payment(sender=wallet.pubkey)
    with pytest.raises(PaymentInvalid, match="64-hex"):
        asyncio.run(_settle(fake, wallet, tx="not-a-hash"))


def test_resubmitting_the_same_payment_succeeds_without_paying_twice(wallet):
    """A client retrying after a dropped response must not be told its payment
    failed -- nor be credited a second window for it."""
    fake = FakeHorizon().payment(sender=wallet.pubkey)
    cfg, store = _config(), WindowStore(":memory:")
    quote = store.issue_quote(wallet.pubkey, cfg.amount, cfg.window_s, now=1000.0)
    fake.tx["memo"] = quote.price_id
    bill = Billing(cfg, store=store, horizon=Horizon(cfg, client_factory=fake.factory()),
                   clock=lambda: 1000.0)

    first = asyncio.run(bill.settle_payment(wallet.pubkey, TX, quote.price_id))
    second = asyncio.run(bill.settle_payment(wallet.pubkey, TX, quote.price_id))
    assert first["credited"] is True and second["credited"] is False
    assert second["settled"] is True
    assert store.row(wallet.pubkey)[1] == 900.0        # one window, not two


def test_a_quote_issued_to_someone_else_cannot_be_redeemed(wallet):
    fake = FakeHorizon().payment(sender=wallet.pubkey)
    cfg, store = _config(), WindowStore(":memory:")
    theirs = store.issue_quote(Wallet().pubkey, cfg.amount, cfg.window_s, now=1000.0)
    bill = Billing(cfg, store=store, horizon=Horizon(cfg, client_factory=fake.factory()),
                   clock=lambda: 1000.0)
    with pytest.raises(PaymentInvalid, match="different identity"):
        asyncio.run(bill.settle_payment(wallet.pubkey, TX, theirs.price_id))


# ------------------------------------------------------- the warm hand-off
def test_on_warm_starts_the_clock_for_a_held_lease(wallet, tmp_path):
    """The `WarmPool` -> `Billing` hand-off, which is where settle-on-grant
    actually happens in the running proxy."""
    cfg = _config(db_path=tmp_path / "b.sqlite")
    bill = Billing(cfg, clock=lambda: 5000.0)
    bill.store.credit_for_tx(TX, wallet.pubkey, 900.0, now=4000.0)
    bill.on_warm([wallet.pubkey])
    assert bill.store.row(wallet.pubkey) == (5900.0, 0.0)
    bill.close()


def test_on_warm_does_not_create_a_database_for_a_deployment_that_never_bills(tmp_path):
    """`on_warm` fires every few seconds while any lease is held. It must not be
    what brings a billing database into existence."""
    path = tmp_path / "unused.sqlite"
    bill = Billing(_config(require_payment=False, db_path=path))
    bill.on_warm(["some-label"])
    assert not path.exists()


def test_the_pool_reports_warmth_to_the_hook():
    """Unconfigured on purpose: `_get_health` short-circuits without a key, so
    this exercises the hand-off and not RunPod."""
    from hvym_img_tools.warm import WarmPool

    seen: list[list[str]] = []
    pool = WarmPool("", "", on_warm=seen.append)

    pool._workers_ready = 1
    pool._notify_warm()
    assert seen == []                       # nobody is holding it; nothing to settle

    asyncio.run(pool.acquire("lease-1", "G-label"))
    pool._workers_ready = 0
    pool._notify_warm()
    assert seen == []                       # leased but still cold: no charge yet

    pool._workers_ready = 1
    pool._notify_warm()
    assert seen == [["G-label"]]


def test_a_hook_that_throws_does_not_take_the_keepalive_down():
    """Warmth is what the artist is paying for. A billing store having a bad day
    must not be able to stop the keepalive that delivers it."""
    from hvym_img_tools.warm import WarmPool

    def explode(_labels):
        raise RuntimeError("billing store is unhappy")

    pool = WarmPool("", "", on_warm=explode)
    asyncio.run(pool.acquire("lease-1", "G-label"))
    pool._workers_ready = 1
    pool._notify_warm()                     # must not raise
    assert pool.owner_of("lease-1") == "G-label"


# ------------------------------------------------------------ over the wire
GOOD_KEY = "proxy-test-key-long-enough-0001"


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Build the app with billing pointed at a temp db and a fake Horizon."""
    monkeypatch.setenv("HVYM_API_KEY", GOOD_KEY)
    monkeypatch.setenv("RUNPOD_API_KEY", "runpod-secret")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "ep123")

    def build(*, horizon=None, clock=None, **cfg_kw):
        cfg = _config(db_path=tmp_path / "billing.sqlite", **cfg_kw)
        fake = horizon or FakeHorizon()
        real = {}

        def factory(*_a, **_kw):
            bill = Billing(
                cfg,
                horizon=Horizon(cfg, client_factory=fake.factory()),
                clock=clock or (lambda: 1000.0),
            )
            real["billing"] = bill
            return bill

        monkeypatch.setattr(proxy_mod, "Billing", factory)
        client = TestClient(proxy_mod.create_app())
        client.headers.update({"X-API-Key": GOOD_KEY})
        return client, real["billing"], fake

    return build


def test_todays_client_is_unaffected_while_the_flags_are_off(wired, monkeypatch):
    """The compat row in docs/X402_BILLING.md §7: with both flags false, an
    unsigned request carrying only X-API-Key still works exactly as before."""
    client, bill, _ = wired(require_identity=False, require_payment=False)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["signed_identity"] is False and resp.json()["payment"] is False


def test_an_unsigned_request_is_401_once_identity_is_enforced(wired):
    client, _, _ = wired(require_identity=True, require_payment=False)
    resp = client.post("/warm", json={"tool": "reangle"})
    assert resp.status_code == 401
    assert "signed identity" in resp.json()["detail"]


def test_an_unpaid_acquire_gets_the_x402_challenge(wired, wallet):
    """The §4.1 body shape, which is the contract the client parses."""
    client, _, _ = wired()
    resp = client.post(
        "/warm", json={"tool": "reangle"},
        headers=wallet.sign("POST", "/warm", "reangle", nonce="1" * 32),
    )
    assert resp.status_code == 402
    challenge = resp.json()["x402"]
    assert challenge["pay_to"] == PAYEE
    assert challenge["issuer"] == USDC_ISSUER
    assert challenge["asset"] == "USDC"
    assert challenge["amount"] == "0.75"
    assert challenge["window_s"] == 900.0
    assert challenge["network"] == "testnet"
    assert challenge["memo"] == challenge["price_id"]
    assert "horizon" in challenge


def test_an_unpaid_tool_call_gets_the_same_challenge(wired, wallet):
    client, _, _ = wired()
    resp = client.post(
        "/tools/mesh", files={"image": ("a.png", b"x", "image/png")},
        headers=wallet.sign("POST", "/tools/mesh", "mesh", nonce="2" * 32),
    )
    assert resp.status_code == 402
    assert resp.json()["x402"]["pay_to"] == PAYEE


def test_a_paid_identity_is_let_through(wired, wallet, monkeypatch):
    client, bill, _ = wired()
    bill.store.credit_for_tx(TX, wallet.pubkey, 900.0, now=1000.0)

    async def fake_acquire(self, lease_id=None, label=""):
        return {"lease_id": "L", "label_seen": label, "state": "warming"}

    monkeypatch.setattr(proxy_mod.WarmPool, "acquire", fake_acquire)
    resp = client.post(
        "/warm", json={"tool": "reangle", "label": "i-am-somebody-else"},
        headers=wallet.sign("POST", "/warm", "reangle", nonce="3" * 32),
    )
    assert resp.status_code == 200
    # The metering label is the verified pubkey, never the one the body claimed.
    assert resp.json()["label_seen"] == wallet.pubkey


def test_the_price_endpoint_is_open_and_issues_no_quote(wired):
    """A UI asking what something costs must not itself be a billable event."""
    client, bill, _ = wired()
    resp = httpx.Client(transport=client._transport, base_url=client.base_url).get("/warm/price")
    assert resp.status_code == 200
    body = resp.json()
    assert body["amount"] == "0.75" and body["pay_to"] == PAYEE and body["enforced"] is True
    assert not bill._has_state          # nothing was written just to answer


def test_warm_status_stays_open_and_unpaid(wired, monkeypatch):
    """GET /warm spends nothing and starts nothing, so payment does not gate it:
    an unpaid client still has to be able to read "cold"."""
    async def fake_status(self):
        return {"state": "cold", "ready": False}

    monkeypatch.setattr(proxy_mod.WarmPool, "status", fake_status)
    client, _, _ = wired()
    resp = httpx.Client(transport=client._transport, base_url=client.base_url).get("/warm")
    assert resp.status_code == 200 and resp.json()["state"] == "cold"


def test_paying_over_http_extends_the_window(wired, wallet, monkeypatch):
    """The end-to-end shape of §4.2: 402 -> pay -> /warm/pay -> entitled."""
    fake = FakeHorizon().payment(sender=wallet.pubkey)
    client, bill, _ = wired(horizon=fake)

    challenge = client.post(
        "/warm", json={"tool": "reangle"},
        headers=wallet.sign("POST", "/warm", "reangle", nonce="4" * 32),
    ).json()["x402"]
    fake.tx["memo"] = challenge["price_id"]

    async def fake_status(self):
        return {"state": "warming", "ready": False}

    monkeypatch.setattr(proxy_mod.WarmPool, "status", fake_status)
    resp = client.post(
        "/warm/pay",
        json={"price_id": challenge["price_id"]},
        headers={**wallet.sign("POST", "/warm/pay", "reangle", nonce="5" * 32), "X-Payment": TX},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["credited"] is True
    assert resp.json()["warm"]["state"] == "warming"
    assert bill.entitled(wallet.pubkey)


def test_a_bad_payment_comes_back_as_402_with_a_fresh_challenge(wired, wallet):
    """A client that paid the wrong thing must be able to fix it in one round
    trip, which means the reason and a payable quote in the same response."""
    fake = FakeHorizon().payment(sender=wallet.pubkey, amount="0.0000001")
    client, _, _ = wired(horizon=fake)
    challenge = client.post(
        "/warm", json={"tool": "reangle"},
        headers=wallet.sign("POST", "/warm", "reangle", nonce="6" * 32),
    ).json()["x402"]
    fake.tx["memo"] = challenge["price_id"]

    resp = client.post(
        "/warm/pay",
        json={"price_id": challenge["price_id"]},
        headers={**wallet.sign("POST", "/warm/pay", "reangle", nonce="7" * 32), "X-Payment": TX},
    )
    assert resp.status_code == 402
    assert "below the quoted" in resp.json()["detail"]
    assert resp.json()["x402"]["pay_to"] == PAYEE


def test_settling_without_an_identity_is_401_not_a_lost_payment(wired):
    """A payment with no identity is money credited to nobody, so this route has
    no un-identified mode even while identity is otherwise only observed."""
    client, _, _ = wired(require_identity=False)
    resp = client.post("/warm/pay", headers={"X-Payment": TX})
    assert resp.status_code == 401


def test_one_artist_cannot_release_anothers_lease(wired, wallet, monkeypatch):
    """Dropping someone else's lease puts out a worker they are still paying for."""
    client, bill, _ = wired()
    bill.store.credit_for_tx(TX, wallet.pubkey, 900.0, now=1000.0)
    monkeypatch.setattr(proxy_mod.WarmPool, "owner_of", lambda self, lid: "G-SOMEBODY-ELSE")
    resp = client.request(
        "DELETE", "/warm", json={"tool": "reangle", "lease_id": "L1"},
        headers=wallet.sign("DELETE", "/warm", "reangle", lease_id="L1", nonce="8" * 32),
    )
    assert resp.status_code == 403


def test_enforcing_payment_with_no_payee_refuses_to_start(monkeypatch):
    """Better a proxy that will not boot than one that 402s every artist with a
    challenge naming no payee -- nobody could satisfy it, and it would read as a
    client bug."""
    monkeypatch.setenv("HVYM_API_KEY", GOOD_KEY)
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "ep")
    monkeypatch.setenv("HVYM_REQUIRE_PAYMENT", "true")
    monkeypatch.delenv("HVYM_PAYEE_ADDRESS", raising=False)
    with pytest.raises(RuntimeError, match="HVYM_PAYEE_ADDRESS"):
        proxy_mod.create_app()
