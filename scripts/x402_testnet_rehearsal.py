"""x402 testnet rehearsal — the one thing the mocked tests cannot prove.

    uv run python scripts/x402_testnet_rehearsal.py

Deliberately NOT part of the pytest suite: it needs network, friendbot, and
about ninety seconds. Run it by hand before flipping `HVYM_REQUIRE_PAYMENT` on
any deployment, and after any change to `billing.Horizon`.

`tests/test_billing.py` mocks Horizon, which covers every *rule* but nothing
about the wire. This runs the whole loop against real Stellar testnet Horizon:
402 -> on-chain payment -> /warm/pay -> settle-on-grant -> a granted lease. What
only a real chain can establish is the shape of what Horizon actually returns
for a `credit_alphanum4` payment operation -- the fields a mock has to guess.

**Nothing here touches mainnet, and no stored identity's secret is read.** Every
keypair is generated in-process and funded by friendbot, including the "USDC"
issuer -- which is the point of minting our own: it makes the two security
checks testable against genuine, successful, correctly-memo'd on-chain payments
that must still be refused.

Those two are the ones worth re-reading when this fails:

  * **the lookalike issuer.** `USDC` is a label anyone can print; only the issuer
    account distinguishes Circle's dollar from a token minted for free. Testnet
    has 200+ accounts issuing something called `USDC`.
  * **the sender binding** (X402_BILLING.md §9 Q3). A stranger paying the right
    payee, asset, amount and memo still cannot fund someone else's window.

A note on the first, because it cost a run: the memo is checked *before* the
operations are, so a fresh database mints a fresh `price_id`, the memo check
fires first, and the issuer check never runs -- the test then passes for
entirely the wrong reason. The lookalike case therefore reuses the original
database on purpose, so the memo matches and the asset is what refuses it.
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
import time
from pathlib import Path

import httpx
from nacl.signing import SigningKey
from stellar_sdk import Asset, Keypair, Network, Server, TransactionBuilder

HORIZON = "https://horizon-testnet.stellar.org"
SCRATCH = Path(tempfile.mkdtemp(prefix="x402-rehearsal-"))
PASS, FAIL = [], []


def step(msg):
    print(f"\n=== {msg}", flush=True)


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + str(detail)) if detail else ''}",
          flush=True)


def fund(pub):
    for attempt in range(3):
        r = httpx.get(f"https://friendbot.stellar.org?addr={pub}", timeout=40)
        if r.status_code < 400:
            return True
        time.sleep(3)
    return False


def b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def sign_headers(kp: Keypair, method, path, tool, lease_id="", nonce=None):
    """Byte-for-byte what RequestSigner.cpp emits."""
    nonce = nonce or os.urandom(16).hex()
    payload = (
        '{"i":"ai-request"'
        f',"lid":"{lease_id}"'
        f',"m":"{method}"'
        f',"n":"{nonce}"'
        f',"p":"{path}"'
        f',"t":{int(time.time())}'
        f',"tool":"{tool}"}}'
    )
    raw = payload.encode()
    sig = SigningKey(kp.raw_secret_key()).sign(raw).signature
    return {"X-Ink-Pubkey": kp.public_key, "X-Ink-Auth": f"{b64u(sig)}.{b64u(raw)}"}


def submit(server, kp, build):
    src = server.load_account(kp.public_key)
    builder = TransactionBuilder(
        src, network_passphrase=Network.TESTNET_NETWORK_PASSPHRASE, base_fee=200
    )
    build(builder)
    tx = builder.set_timeout(90).build()
    tx.sign(kp)
    return server.submit_transaction(tx)


# --------------------------------------------------------------- chain setup
step("Generating three throwaway testnet accounts")
artist = Keypair.random()      # the wallet that signs requests AND pays
payee = Keypair.random()       # HEAVYMETA's receiving account
issuer = Keypair.random()      # our own testnet "USDC" issuer
for name, kp in (("artist", artist), ("payee", payee), ("issuer", issuer)):
    print(f"  {name:7} {kp.public_key}", flush=True)

step("Funding them with friendbot")
for name, kp in (("artist", artist), ("payee", payee), ("issuer", issuer)):
    check(f"friendbot funded {name}", fund(kp.public_key))
if FAIL:
    sys.exit("friendbot failed; cannot continue")

server = Server(HORIZON)
USDC = Asset("USDC", issuer.public_key)

step("Trustlines + issuing test USDC")
submit(server, payee, lambda b: b.append_change_trust_op(asset=USDC))
submit(server, artist, lambda b: b.append_change_trust_op(asset=USDC))
submit(server, issuer, lambda b: b.append_payment_op(
    destination=artist.public_key, asset=USDC, amount="100"))
bal = {
    (x.get("asset_code"), x.get("asset_issuer")): x["balance"]
    for x in server.accounts().account_id(artist.public_key).call()["balances"]
}
check("artist holds test USDC", bal.get(("USDC", issuer.public_key)) is not None, bal)

# ------------------------------------------------------------------ the app
step("Starting the proxy with payment ENFORCED against testnet")
db = SCRATCH / "rehearsal.sqlite"
db.unlink(missing_ok=True)
os.environ.update(
    HVYM_API_KEY="rehearsal-key-long-enough-01",
    RUNPOD_API_KEY="fake-not-used",
    RUNPOD_ENDPOINT_ID="ep-fake",
    HVYM_REQUIRE_SIGNED_IDENTITY="true",
    HVYM_REQUIRE_PAYMENT="true",
    HVYM_STELLAR_NETWORK="testnet",
    HVYM_SETTLE_ASSET="USDC",
    HVYM_USDC_ISSUER=issuer.public_key,
    HVYM_PAYEE_ADDRESS=payee.public_key,
    HVYM_AI_WINDOW_S="900",
    HVYM_AI_PRICE_PER_MIN="0.05",
    HVYM_BILLING_DB=str(db),
)

from fastapi.testclient import TestClient          # noqa: E402
from hvym_img_tools import proxy as proxy_mod      # noqa: E402
from hvym_img_tools.warm import WarmPool           # noqa: E402


async def fake_health(self, force=False):
    """Stand in for RunPod only. Reports a warm worker so settle-on-grant fires
    -- that transition is the thing under test, RunPod is not."""
    self._workers_ready = 1
    self._notify_warm()
    return 1


async def no_ping(self):
    return None


WarmPool._get_health = fake_health
WarmPool._ping = no_ping

client = TestClient(proxy_mod.create_app())
client.headers.update({"X-API-Key": "rehearsal-key-long-enough-01"})

step("GET /warm/price is open and quotes the window")
price = client.get("/warm/price").json()
check("price is 0.75 USDC / 900s", price["amount"] == "0.75" and price["window_s"] == 900.0, price)
check("price names the payee", price["pay_to"] == payee.public_key)

step("POST /warm unpaid -> 402 with a payable challenge")
r = client.post("/warm", json={"tool": "reangle"},
                headers=sign_headers(artist, "POST", "/warm", "reangle"))
check("unpaid acquire is 402", r.status_code == 402, r.status_code)
ch = r.json().get("x402", {})
check("challenge carries pay_to/amount/memo",
      ch.get("pay_to") == payee.public_key and ch.get("amount") == "0.75" and ch.get("memo"))
check("challenge names our issuer", ch.get("issuer") == issuer.public_key)
price_id = ch.get("price_id")
print(f"  price_id/memo = {price_id}", flush=True)

step("Paying it, for real, on testnet")
resp = submit(server, artist, lambda b: b.add_text_memo(price_id).append_payment_op(
    destination=payee.public_key, asset=USDC, amount="0.75"))
tx_hash = resp["hash"]
check("payment landed on-chain", resp.get("successful", False), tx_hash)
print(f"  https://stellar.expert/explorer/testnet/tx/{tx_hash}", flush=True)

step("POST /warm/pay settles it")
r = client.post("/warm/pay", json={"price_id": price_id, "tool": "reangle"},
                headers={**sign_headers(artist, "POST", "/warm/pay", "reangle"),
                         "X-Payment": tx_hash})
check("settle returns 200", r.status_code == 200, r.text[:300])
if r.status_code == 200:
    body = r.json()
    check("credited", body.get("credited") is True, body)
    check("amount read back from chain", body.get("amount") == "0.75", body.get("amount"))

step("The lease is now granted")
r = client.post("/warm", json={"tool": "reangle"},
                headers=sign_headers(artist, "POST", "/warm", "reangle"))
check("paid acquire is 200", r.status_code == 200, r.status_code)
check("lease labelled with the wallet", r.json().get("lease_id") is not None)

step("Settle-on-grant: the window started when the worker warmed")
from hvym_img_tools.billing import Billing, BillingConfig   # noqa: E402
peek = Billing(BillingConfig.from_env())
paid_through, credit = peek.store.row(artist.public_key)
check("credit consumed", credit == 0.0, credit)
check("paid_through ~900s out", 800 < (paid_through - time.time()) <= 900,
      round(paid_through - time.time(), 1))
peek.close()

step("Replaying the same tx does not buy a second window")
r = client.post("/warm/pay", json={"price_id": price_id, "tool": "reangle"},
                headers={**sign_headers(artist, "POST", "/warm/pay", "reangle"),
                         "X-Payment": tx_hash})
check("replay is 200", r.status_code == 200, r.status_code)
check("replay does not re-credit", r.json().get("credited") is False, r.text[:200])

step("A real, successful, correctly-memo'd payment is REFUSED under a lookalike issuer")
# Same database on purpose, so the ORIGINAL price_id is still a live quote and
# the memo check passes. Otherwise a fresh db mints a new price_id, the memo
# check fires first, and the issuer check -- the one thing being tested -- never
# runs. (It passed for exactly that wrong reason on the first attempt.)
os.environ["HVYM_USDC_ISSUER"] = Keypair.random().public_key   # a lookalike USDC
lookalike = TestClient(proxy_mod.create_app())
lookalike.headers.update({"X-API-Key": "rehearsal-key-long-enough-01"})
r = lookalike.post("/warm/pay", json={"price_id": price_id, "tool": "reangle"},
                   headers={**sign_headers(artist, "POST", "/warm/pay", "reangle"),
                            "X-Payment": tx_hash})
detail = r.json().get("detail", "")
check("lookalike issuer refuses the payment", r.status_code == 402, r.status_code)
check("refused on the ASSET, not the memo", "not in USDC" in detail, detail)
os.environ["HVYM_USDC_ISSUER"] = issuer.public_key

step("A payment from a different wallet is REFUSED (sender binding, \u00a79 Q3)")
stranger = Keypair.random()
check("friendbot funded stranger", fund(stranger.public_key))
submit(server, stranger, lambda b: b.append_change_trust_op(asset=USDC))
submit(server, issuer, lambda b: b.append_payment_op(
    destination=stranger.public_key, asset=USDC, amount="10"))

# A brand new identity, so it has its own live quote to pay against.
newcomer = Keypair.random()
bound = TestClient(proxy_mod.create_app())
bound.headers.update({"X-API-Key": "rehearsal-key-long-enough-01"})
r = bound.post("/warm", json={"tool": "reangle"},
               headers=sign_headers(newcomer, "POST", "/warm", "reangle"))
pid3 = r.json()["x402"]["price_id"]

# The stranger pays the newcomer's quote correctly -- right payee, right asset,
# right amount, right memo. Only the sender is wrong.
resp3 = submit(server, stranger, lambda b: b.add_text_memo(pid3).append_payment_op(
    destination=payee.public_key, asset=USDC, amount="0.75"))
check("third-party payment landed on-chain", resp3.get("successful", False), resp3["hash"])
r = bound.post("/warm/pay", json={"price_id": pid3, "tool": "reangle"},
               headers={**sign_headers(newcomer, "POST", "/warm/pay", "reangle"),
                        "X-Payment": resp3["hash"]})
detail = r.json().get("detail", "")
check("a stranger cannot fund someone else's window", r.status_code == 402, r.status_code)
check("refused on the SENDER", "not sent by the identity" in detail, detail)

print("\n" + "=" * 60)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  failed:", ", ".join(FAIL))
print("=" * 60)
sys.exit(1 if FAIL else 0)
