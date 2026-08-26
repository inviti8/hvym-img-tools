#!/usr/bin/env bash
# Pull a file back from the pod over proxy SSH. Base64 goes straight to a local file
# (never stdout). Markers are split in the source text (B64S""TART) so the echoed
# command line cannot match them -- only the real remote output does.
set -euo pipefail
SRC="$1"; DST="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$(dirname "$DST")"
RAW="$(mktemp)"
TMO="${TMO:-600}" bash "$HERE/podrun.sh" "echo B64S\"\"TART; base64 $SRC; echo B64E\"\"ND" > "$RAW" 2>/dev/null
sed -n '/B64START$/,/B64END$/{//!p}' "$RAW" | grep -E '^[A-Za-z0-9+/]+={0,2}$' | base64 -d > "$DST" 2>/dev/null || true
REMOTE_SHA=$(TMO=120 bash "$HERE/podrun.sh" "sha256sum $SRC" | grep -oE '[0-9a-f]{64}' | head -1)
LOCAL_SHA=$(sha256sum "$DST" | cut -d' ' -f1)
rm -f "$RAW"
if [ "$LOCAL_SHA" = "$REMOTE_SHA" ]; then echo "OK   $(basename "$SRC") -> $DST ($(stat -c%s "$DST") bytes, sha verified)"
else echo "FAIL $(basename "$SRC") ($(stat -c%s "$DST") bytes) local=$LOCAL_SHA remote=$REMOTE_SHA"; fi
