"""Signed wallet identity — the verifier half of Inkternity's `RequestSigner`.

**Why this exists.** `core.auth` is a shared secret shipped inside a desktop
binary: extractable by anyone, therefore a spend-control measure and not an
identity boundary (see its own docstring, and docs/AUTH.md §Upgrade path steps 2
and 4). This module is those two steps. Each request carries an ed25519
signature made by the artist's own wallet key, so the proxy learns *which
wallet* is asking — which is the thing billing can be attached to.

**The wire contract is the client's, and it is byte-exact.** The authority is
`../infinipaint/src/AI/RequestSigner.hpp`; docs/X402_BILLING.md §2 restates it.
Two headers ride on every `POST /warm`, `DELETE /warm`, `POST /warm/pay` and
`POST /tools/{name}`::

    X-Ink-Pubkey: <G... Stellar strkey>
    X-Ink-Auth:   <base64url(64-byte sig)> "." <base64url(payload)>

`payload` is compact JSON with alphabetical keys and no spaces::

    {"i":"ai-request","lid":"<lease_id|>","m":"<METHOD>","n":"<nonce-hex>",
     "p":"<path>","t":<unix-seconds>,"tool":"<tool>"}

The client hand-builds those bytes precisely so a verifier does not have to
agree with it about JSON serialization. **We therefore verify the signature over
the bytes as received and never re-serialize them.** Re-encoding the parsed dict
would reintroduce exactly the ambiguity the client went out of its way to avoid.

**Why no `stellar_sdk`.** All we do here is decode a strkey and check an ed25519
signature — 40 lines against a dependency tree (aiohttp, requests, mnemonic,
toml) that would land in the proxy image, whose whole design constraint is being
small enough to share a cheap always-on box (docker/Dockerfile.proxy). PyNaCl
alone is enough. `tests/test_billing.py` round-trips this against the real
`stellar_sdk` so the shortcut stays honest.

Body bytes are deliberately *not* signed: the `/tools/*` body is a
curl-generated multipart the client never holds as one buffer. See
docs/X402_BILLING.md §2 "Body is intentionally NOT signed" for why that is
bounded on a pay-only service.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import struct
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: Headers the client sends. Named for Inkternity because that is who signs.
PUBKEY_HEADER = "X-Ink-Pubkey"
AUTH_HEADER = "X-Ink-Auth"

#: The payload's `i` field. A domain tag, so a signature minted for one purpose
#: (a C2PA WireToken, a subscription token — same envelope shape) can never be
#: replayed as an AI request.
INTENT = "ai-request"

#: Freshness window, and the TTL of the replay cache. Both, deliberately: a
#: nonce only has to be remembered for as long as a signature bearing it would
#: still be accepted.
SIGN_WINDOW_S = float(os.environ.get("HVYM_SIGN_WINDOW_S", "120"))

#: Stellar strkey version byte for an ed25519 public key ('G...').
_VERSION_ED25519_PUBLIC = 6 << 3

#: A 32-byte key + 1 version byte + 2 checksum bytes, base32'd with no padding.
_STRKEY_LEN = 56

#: Bounds on what we will even attempt to decode, so a hostile header cannot
#: make us do work proportional to its size.
_MAX_AUTH_HEADER = 4096


class IdentityError(Exception):
    """A signature we would not accept, and the check that rejected it.

    The message names the failed check and nothing else — no key material, no
    header echo — so it is safe both in a log line and as a 401 `detail`.
    """


# ------------------------------------------------------------------- strkey
def _crc16_xmodem(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def decode_ed25519_public_key(strkey: str) -> bytes:
    """`G...` strkey -> the raw 32-byte ed25519 public key.

    Checks the version byte and the CRC16 rather than trusting the length: a
    typo'd address that happens to be 56 characters must not become a distinct
    billing identity that no one can ever pay from.
    """
    if not strkey or len(strkey) != _STRKEY_LEN or not strkey.startswith("G"):
        raise IdentityError("pubkey is not a 56-character G... strkey")
    try:
        raw = base64.b32decode(strkey.encode("ascii"), casefold=False)
    except (binascii.Error, ValueError) as exc:
        raise IdentityError("pubkey is not valid base32") from exc
    if len(raw) != 35:
        raise IdentityError("pubkey decodes to the wrong length")
    version, payload, checksum = raw[0], raw[1:33], raw[33:]
    if version != _VERSION_ED25519_PUBLIC:
        raise IdentityError("pubkey is not an ed25519 public key")
    if struct.unpack("<H", checksum)[0] != _crc16_xmodem(raw[:33]):
        raise IdentityError("pubkey checksum does not match")
    return payload


def encode_ed25519_public_key(raw: bytes) -> str:
    """Inverse of :func:`decode_ed25519_public_key`. Used by tests and tooling;
    the proxy itself only ever decodes."""
    if len(raw) != 32:
        raise IdentityError("an ed25519 public key is 32 bytes")
    body = bytes([_VERSION_ED25519_PUBLIC]) + raw
    return base64.b32encode(body + struct.pack("<H", _crc16_xmodem(body))).decode("ascii")


def _b64u_decode(text: str, *, what: str) -> bytes:
    """URL-safe base64, padding optional — the client emits none."""
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError) as exc:
        raise IdentityError(f"{what} is not valid base64url") from exc


# ------------------------------------------------------------------ identity
@dataclass(frozen=True, slots=True)
class SignedIdentity:
    """A request whose signature checked out.

    `pubkey` is the billing identity *and* the metering label — the proxy sets
    `Lease.label` from it and never from the request body, so attribution cannot
    be spoofed by a client that simply claims a different label.
    """

    pubkey: str
    tool: str
    lease_id: str
    nonce: str
    issued_at: int


class IdentityVerifier:
    """Verifies signatures and refuses replays, over one freshness window.

    In-process state, deliberately. The replay cache only has to outlive the
    freshness window (120 s), and a proxy restart invalidates every signature
    older than that anyway, so persisting it would buy nothing. If this ever
    runs multi-instance behind a load balancer, that changes — a nonce seen by
    one instance is unseen by its sibling — and the cache moves to the shared
    store next to `paid_through`.
    """

    def __init__(self, *, window_s: float = SIGN_WINDOW_S, clock=time.time) -> None:
        self.window_s = float(window_s)
        self._clock = clock
        self._seen: dict[tuple[str, str], float] = {}

    @staticmethod
    def presented(pubkey_header: str | None, auth_header: str | None) -> bool:
        """Whether the client attempted to sign at all.

        The distinction matters during the log-only rollout: an unsigned request
        from today's client is expected, while a *malformed* signature from a
        client that meant to sign is a bug worth seeing in the logs.
        """
        return bool(pubkey_header and auth_header)

    def _prune(self, now: float) -> None:
        if len(self._seen) < 512:  # cheap guard; the common case is a handful
            expired = [k for k, exp in self._seen.items() if exp <= now]
        else:
            expired = [k for k, exp in list(self._seen.items()) if exp <= now]
        for key in expired:
            del self._seen[key]

    def verify(
        self,
        *,
        pubkey_header: str | None,
        auth_header: str | None,
        method: str,
        path: str,
        tool: str,
        lease_id: str = "",
    ) -> SignedIdentity:
        """Run every check in docs/X402_BILLING.md §2. Raises `IdentityError`.

        The order is not arbitrary: the nonce is consumed *last*, only once the
        request is otherwise fully accepted. Burning it earlier would let an
        unsigned or malformed request poison a nonce the legitimate client is
        about to use.
        """
        if not self.presented(pubkey_header, auth_header):
            raise IdentityError(f"missing {PUBKEY_HEADER} / {AUTH_HEADER}")
        assert auth_header is not None and pubkey_header is not None
        if len(auth_header) > _MAX_AUTH_HEADER:
            raise IdentityError("auth header is implausibly large")

        # 1. envelope
        sig_b64, dot, payload_b64 = auth_header.partition(".")
        if not dot or not sig_b64 or not payload_b64:
            raise IdentityError("auth header is not <sig>.<payload>")
        signature = _b64u_decode(sig_b64, what="signature")
        payload_bytes = _b64u_decode(payload_b64, what="payload")
        if len(signature) != 64:
            raise IdentityError("signature is not 64 bytes")
        if not payload_bytes:
            raise IdentityError("payload is empty")

        # 2-3. the signature, over the bytes exactly as received
        verify_key = decode_ed25519_public_key(pubkey_header.strip())
        try:
            from nacl.exceptions import BadSignatureError  # noqa: PLC0415 - keeps
            from nacl.signing import VerifyKey            # core CPU-light on import

            VerifyKey(verify_key).verify(payload_bytes, signature)
        except ImportError as exc:  # pragma: no cover - deployment error
            raise IdentityError("ed25519 verification is unavailable (PyNaCl missing)") from exc
        except BadSignatureError as exc:
            raise IdentityError("signature does not verify for this pubkey") from exc

        # 4. bind the payload to the request it arrived on
        try:
            claims = json.loads(payload_bytes)
        except ValueError as exc:
            raise IdentityError("payload is not JSON") from exc
        if not isinstance(claims, dict):
            raise IdentityError("payload is not a JSON object")
        if claims.get("i") != INTENT:
            raise IdentityError("payload is not an ai-request")
        for field, expected, label in (
            ("m", method.upper(), "method"),
            ("p", path, "path"),
            ("tool", tool, "tool"),
            ("lid", lease_id or "", "lease_id"),
        ):
            if str(claims.get(field, "")) != expected:
                raise IdentityError(f"signed {label} does not match the request")

        # 5. freshness
        try:
            issued_at = int(claims["t"])
        except (KeyError, TypeError, ValueError) as exc:
            raise IdentityError("payload has no usable timestamp") from exc
        now = self._clock()
        if abs(now - issued_at) > self.window_s:
            raise IdentityError(f"signature is outside the {self.window_s:.0f}s freshness window")

        # 6. replay
        nonce = str(claims.get("n") or "")
        if not nonce:
            raise IdentityError("payload has no nonce")
        pubkey = pubkey_header.strip()
        self._prune(now)
        if (pubkey, nonce) in self._seen:
            raise IdentityError("nonce has already been used")
        self._seen[(pubkey, nonce)] = now + self.window_s

        return SignedIdentity(
            pubkey=pubkey,
            tool=str(claims.get("tool") or ""),
            lease_id=str(claims.get("lid") or ""),
            nonce=nonce,
            issued_at=issued_at,
        )
