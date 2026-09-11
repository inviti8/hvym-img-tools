"""x402 end-to-end server — the app-side counterpart to x402_testnet_rehearsal.

    uv run --extra server python scripts/x402_e2e_server.py [port] [configdir]

Unlike the rehearsal (which drives the proxy in-process via TestClient), this
serves the proxy over REAL HTTP on testnet with payment ENFORCED, so the actual
compiled Inkternity client can hit it:

    inkternity --x402-selftest http://127.0.0.1:<port> <api-key> <configdir> <out>

It mints throwaway testnet accounts (issuer / payee / artist), funds them via
friendbot, gives the artist a USDC trustline + balance, and writes the artist
keypair as <configdir>/inkternity_dev_keys.json — the wallet the app will sign
and pay from. RunPod health is faked (as in the rehearsal) so settle-on-grant
can fire without a real GPU. Nothing here touches mainnet.

When ready it writes <configdir>/e2e_ready.json {port, api_key, issuer, payee,
artist} and then serves until Ctrl-C.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
from stellar_sdk import Asset, Keypair, Network, Server, TransactionBuilder

HORIZON = "https://horizon-testnet.stellar.org"
API_KEY = "e2e-key-long-enough-0123456789"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
CONFIG_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(os.getcwd()) / "_x402_e2e"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def log(m):
    print(m, flush=True)


def fund(pub):
    for _ in range(3):
        if httpx.get(f"https://friendbot.stellar.org?addr={pub}", timeout=40).status_code < 400:
            return True
        time.sleep(3)
    return False


def submit(server, kp, build):
    src = server.load_account(kp.public_key)
    b = TransactionBuilder(src, network_passphrase=Network.TESTNET_NETWORK_PASSPHRASE, base_fee=200)
    build(b)
    tx = b.set_timeout(90).build()
    tx.sign(kp)
    return server.submit_transaction(tx)


log("=== minting throwaway testnet accounts (issuer / payee / artist)")
issuer, payee, artist = Keypair.random(), Keypair.random(), Keypair.random()
for name, kp in (("issuer", issuer), ("payee", payee), ("artist", artist)):
    ok = fund(kp.public_key)
    log(f"  friendbot {name:6} {kp.public_key}  {'ok' if ok else 'FAILED'}")
    if not ok:
        sys.exit("friendbot failed")

server = Server(HORIZON)
USDC = Asset("USDC", issuer.public_key)

log("=== trustlines + issuing test USDC to the artist wallet")
submit(server, payee, lambda b: b.append_change_trust_op(asset=USDC))
submit(server, artist, lambda b: b.append_change_trust_op(asset=USDC))
submit(server, issuer, lambda b: b.append_payment_op(destination=artist.public_key, asset=USDC, amount="10"))
bal = {(x.get("asset_code"), x.get("asset_issuer")): x["balance"]
       for x in server.accounts().account_id(artist.public_key).call()["balances"]}
log(f"  artist USDC balance = {bal.get(('USDC', issuer.public_key))}")

log("=== writing the artist wallet as the app's DevKeys file")
devkeys = CONFIG_DIR / "inkternity_dev_keys.json"
devkeys.write_text(json.dumps({"app_pub": artist.public_key, "app_secret": artist.secret}, indent=2))
log(f"  {devkeys}")

log("=== configuring the proxy: testnet, payment ENFORCED, fake RunPod")
db = CONFIG_DIR / "e2e_billing.sqlite"
db.unlink(missing_ok=True)
os.environ.update(
    HVYM_API_KEY=API_KEY,
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

from hvym_img_tools import proxy as proxy_mod   # noqa: E402
from hvym_img_tools.warm import WarmPool         # noqa: E402


async def fake_health(self, force=False):
    self._workers_ready = 1
    self._notify_warm()
    return 1


async def no_ping(self):
    return None


WarmPool._get_health = fake_health
WarmPool._ping = no_ping

(CONFIG_DIR / "e2e_ready.json").write_text(json.dumps({
    "port": PORT, "api_key": API_KEY,
    "issuer": issuer.public_key, "payee": payee.public_key, "artist": artist.public_key,
}, indent=2))

log(f"=== serving on http://127.0.0.1:{PORT}  (config dir: {CONFIG_DIR})")
import uvicorn   # noqa: E402
uvicorn.run(proxy_mod.create_app(), host="127.0.0.1", port=PORT, log_level="warning")
