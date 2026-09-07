"""The x402 loop against the REAL Centre-issued testnet USDC.

    uv run python scripts/x402_testnet_real_usdc.py [--payer <stellar-cli-identity>]

`x402_testnet_rehearsal.py` is the one to run routinely: it is self-contained,
mints its own issuer, and proves the negative cases. This is its companion for
the last mile of realism — it pays with the asset Circle actually issues on
testnet, so the Horizon record the verifier parses is the same shape mainnet
will hand it, from the same issuer software, with a real `home_domain` behind it.

**Prerequisites, and they are the same ones mainnet has:**

  * a stellar-cli identity holding testnet USDC (default `testnet-deployer`)
  * that identity, and the payee, each need a **trustline** to the issuer below

The payee here is generated and given its trustline automatically. The payer's
is your problem, and is exactly the step that bites: an account with no
trustline cannot receive an issued asset at all, so a send to it fails with
`op_no_trust` at the *sender's* end and never appears on the recipient. That is
not a hypothetical — it is what happened the first time testnet USDC was sent to
`testnet-deployer` for this script.

The payer's secret is read from the stellar CLI into memory to sign the
`X-Ink-Auth` headers (the client signs with the same ed25519 key that owns the
wallet, so the two cannot be separated). It is never printed or written down.
Point `--payer` at a throwaway identity; never at anything holding mainnet value.
"""
from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from nacl.signing import SigningKey
from stellar_sdk import Asset, Keypair, Network, Server, TransactionBuilder

#: Centre's testnet USDC. Verified on-chain: this account's `home_domain` is
#: `centre.io`. Note it is NOT the mainnet issuer -- that one is a different
#: account whose home_domain is `circle.com`. Getting these two confused means
#: quoting a price nobody can pay.
TESTNET_USDC_ISSUER = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
HORIZON = "https://horizon-testnet.stellar.org"

PASS, FAIL = [], []


def step(msg):
    print(f"\n=== {msg}", flush=True)


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + str(detail)) if detail else ''}",
          flush=True)


def b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def sign_headers(kp, method, path, tool, lease_id=""):
    """Byte-for-byte what ../infinipaint/src/AI/RequestSigner.cpp emits."""
    payload = (
        '{"i":"ai-request"'
        f',"lid":"{lease_id}"'
        f',"m":"{method}"'
        f',"n":"{os.urandom(16).hex()}"'
        f',"p":"{path}"'
        f',"t":{int(time.time())}'
        f',"tool":"{tool}"}}'
    )
    raw = payload.encode()
    return {
        "X-Ink-Pubkey": kp.public_key,
        "X-Ink-Auth": f"{b64u(SigningKey(kp.raw_secret_key()).sign(raw).signature)}.{b64u(raw)}",
    }


def submit(server, kp, build):
    src = server.load_account(kp.public_key)
    builder = TransactionBuilder(
        src, network_passphrase=Network.TESTNET_NETWORK_PASSPHRASE, base_fee=200
    )
    build(builder)
    tx = builder.set_timeout(90).build()
    tx.sign(kp)
    return server.submit_transaction(tx)


def load_payer(identity: str) -> Keypair:
    """Read a stellar-cli identity's secret into memory, and nowhere else."""
    exe = shutil.which("stellar")
    if not exe:
        sys.exit("the stellar CLI is not on PATH")
    done = subprocess.run([exe, "keys", "secret", identity],
                          capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"could not read identity {identity!r} (is it in `stellar keys ls`?)")
    return Keypair.from_secret(done.stdout.strip())


parser = argparse.ArgumentParser()
parser.add_argument("--payer", default="testnet-deployer")
args = parser.parse_args()

server = Server(HORIZON)
USDC = Asset("USDC", TESTNET_USDC_ISSUER)

step(f"Payer: stellar-cli identity {args.payer!r}")
artist = load_payer(args.payer)
print(f"  {artist.public_key}", flush=True)
balances = {
    (b.get("asset_code"), b.get("asset_issuer")): b["balance"]
    for b in server.accounts().account_id(artist.public_key).call()["balances"]
}
held = balances.get(("USDC", TESTNET_USDC_ISSUER))
check("payer holds real testnet USDC", held is not None and float(held) >= 1, held)
if FAIL:
    sys.exit(
        f"\nFund it first: establish a trustline, then send USDC.\n"
        f"  stellar tx new change-trust --source-account {args.payer} "
        f"--line USDC:{TESTNET_USDC_ISSUER} --network testnet"
    )

step("Generating the payee and giving it a trustline")
payee = Keypair.random()
print(f"  {payee.public_key}", flush=True)
r = httpx.get(f"https://friendbot.stellar.org?addr={payee.public_key}", timeout=45)
check("friendbot funded the payee", r.status_code < 400, r.status_code)
submit(server, payee, lambda b: b.append_change_trust_op(asset=USDC))
check("payee can now receive USDC", any(
    x.get("asset_issuer") == TESTNET_USDC_ISSUER
    for x in server.accounts().account_id(payee.public_key).call()["balances"]
))

step("Starting the proxy with payment ENFORCED against real testnet USDC")
os.environ.update(
    HVYM_API_KEY="rehearsal-key-long-enough-01",
    RUNPOD_API_KEY="fake-not-used",
    RUNPOD_ENDPOINT_ID="ep-fake",
    HVYM_REQUIRE_SIGNED_IDENTITY="true",
    HVYM_REQUIRE_PAYMENT="true",
    HVYM_STELLAR_NETWORK="testnet",
    HVYM_SETTLE_ASSET="USDC",
    HVYM_USDC_ISSUER=TESTNET_USDC_ISSUER,
    HVYM_PAYEE_ADDRESS=payee.public_key,
    HVYM_AI_WINDOW_S="900",
    HVYM_AI_PRICE_PER_MIN="0.05",
    HVYM_BILLING_DB=str(Path(tempfile.mkdtemp(prefix="x402-real-")) / "billing.sqlite"),
)

from fastapi.testclient import TestClient          # noqa: E402
from hvym_img_tools import proxy as proxy_mod      # noqa: E402
from hvym_img_tools.warm import WarmPool           # noqa: E402


async def fake_health(self, force=False):
    """Stands in for RunPod only -- the settle-on-grant transition is what is
    under test here, not the GPU."""
    self._workers_ready = 1
    self._notify_warm()
    return 1


WarmPool._get_health = fake_health
WarmPool._ping = lambda self: _noop()


async def _noop():
    return None


client = TestClient(proxy_mod.create_app())
client.headers.update({"X-API-Key": "rehearsal-key-long-enough-01"})

step("Unpaid acquire -> 402")
r = client.post("/warm", json={"tool": "reangle"},
                headers=sign_headers(artist, "POST", "/warm", "reangle"))
check("402", r.status_code == 402, r.status_code)
ch = r.json().get("x402", {})
check("challenge names Centre's issuer", ch.get("issuer") == TESTNET_USDC_ISSUER)
price_id = ch.get("price_id")
print(f"  memo = {price_id}   amount = {ch.get('amount')} USDC", flush=True)

step("Paying 0.75 real testnet USDC")
resp = submit(server, artist, lambda b: b.add_text_memo(price_id).append_payment_op(
    destination=payee.public_key, asset=USDC, amount=ch["amount"]))
check("landed on-chain", resp.get("successful", False), resp["hash"])
print(f"  https://stellar.expert/explorer/testnet/tx/{resp['hash']}", flush=True)

step("What Horizon actually returns for that payment operation")
ops = httpx.get(f"{HORIZON}/transactions/{resp['hash']}/operations?limit=200", timeout=30).json()
op = next(o for o in ops["_embedded"]["records"] if o["type"] == "payment")
for field in ("type", "asset_type", "asset_code", "asset_issuer", "from", "to", "amount"):
    print(f"  {field:12} {op.get(field)}", flush=True)
check("asset_type is credit_alphanum4", op.get("asset_type") == "credit_alphanum4")
check("the fields billing.Horizon reads are all present",
      all(op.get(f) is not None for f in ("asset_code", "asset_issuer", "from", "to", "amount")))

step("Settling it")
r = client.post("/warm/pay", json={"price_id": price_id, "tool": "reangle"},
                headers={**sign_headers(artist, "POST", "/warm/pay", "reangle"),
                         "X-Payment": resp["hash"]})
check("settle returns 200", r.status_code == 200, r.text[:300])
check("credited", r.status_code == 200 and r.json().get("credited") is True)

step("The lease is granted")
r = client.post("/warm", json={"tool": "reangle"},
                headers=sign_headers(artist, "POST", "/warm", "reangle"))
check("paid acquire is 200", r.status_code == 200, r.status_code)

print("\n" + "=" * 60)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  failed:", ", ".join(FAIL))
print("=" * 60)
sys.exit(1 if FAIL else 0)
