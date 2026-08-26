#!/usr/bin/env bash
# Send a local file over the proxy-SSH stdin channel as base64 (no scp without a public IP).
# Wrapped at 76 cols so the PTY's canonical-mode ~4096B line limit is never hit.
set -euo pipefail
SRC="$1"; DST="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOCAL_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
SCRIPT=$(printf 'mkdir -p "$(dirname %s)"\ncat > /tmp/xfer.b64 <<'"'"'B64EOF'"'"'\n%s\nB64EOF\nbase64 -d /tmp/xfer.b64 > %s\necho "REMOTE_SHA=$(sha256sum %s | cut -d" " -f1)"\necho "REMOTE_SIZE=$(stat -c%%s %s)"\n' \
  "$DST" "$(base64 "$SRC")" "$DST" "$DST" "$DST")
OUT=$(TMO="${TMO:-300}" bash "$HERE/podrun.sh" "$SCRIPT" | grep -oE 'REMOTE_(SHA|SIZE)=[0-9a-f]+')
REMOTE_SHA=$(echo "$OUT" | grep REMOTE_SHA | cut -d= -f2)
if [ "$LOCAL_SHA" = "$REMOTE_SHA" ]; then
  echo "OK  $SRC -> $DST  ($(stat -c%s "$SRC") bytes, sha verified)"
else
  echo "FAIL $SRC -> $DST"; echo "  local =$LOCAL_SHA"; echo "  remote=$REMOTE_SHA"; echo "$OUT"; exit 1
fi
