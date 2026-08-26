"""Shared-secret API-key auth.

**Threat model — read this before relying on it.** A key shipped inside a
distributed desktop app is *extractable*: anyone can pull it from the binary or
read it off the wire with a TLS-intercepting proxy. This is not an identity
boundary. What it does buy:

- keeps casual/accidental and drive-by traffic off a **GPU that bills per second**
- gives you a revocation lever (rotate the key, ship an update)
- makes "who is allowed to call this" an explicit, testable decision

For a real boundary you want per-user credentials issued at runtime, or a proxy
that authenticates users and holds the service key server-side. See §Upgrade path
in docs/AUTH.md.

Deliberately dependency-free and CPU-only so it lives in `core` like everything
else.
"""
from __future__ import annotations

import hmac
import logging
import secrets
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: Header the client is expected to send. `Authorization: Bearer <key>` also works.
API_KEY_HEADER = "X-API-Key"

#: Below this, a key is too weak to be worth having.
MIN_KEY_LENGTH = 16


@dataclass(slots=True)
class ApiKeyAuth:
    """Verifies a request key against a set of accepted keys.

    Multiple keys are supported so a key can be rotated without downtime: add
    the new one, ship the client update, then drop the old one.
    """

    keys: frozenset[str]

    @property
    def enabled(self) -> bool:
        return bool(self.keys)

    @classmethod
    def from_keys(cls, keys: tuple[str, ...] | list[str]) -> "ApiKeyAuth":
        cleaned = {k.strip() for k in keys if k and k.strip()}
        for key in cleaned:
            if len(key) < MIN_KEY_LENGTH:
                log.warning(
                    "API key of length %d is shorter than the %d-char minimum; "
                    "generate one with `python -m hvym_img_tools.core.auth`",
                    len(key), MIN_KEY_LENGTH,
                )
        return cls(keys=frozenset(cleaned))

    def verify(self, presented: str | None) -> bool:
        """Constant-time check.

        `hmac.compare_digest` against every key, with no early exit, so response
        timing does not leak how much of a key was correct.
        """
        if not self.enabled:
            return True
        if not presented:
            return False
        ok = False
        for key in self.keys:
            if hmac.compare_digest(presented, key):
                ok = True
        return ok


def extract_key(header_value: str | None, authorization: str | None) -> str | None:
    """Accept either `X-API-Key: <key>` or `Authorization: Bearer <key>`."""
    if header_value:
        return header_value.strip()
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    return None


def generate_key(nbytes: int = 32) -> str:
    """A key suitable for HVYM_API_KEY."""
    return secrets.token_urlsafe(nbytes)


if __name__ == "__main__":  # pragma: no cover - operator convenience
    print(generate_key())
