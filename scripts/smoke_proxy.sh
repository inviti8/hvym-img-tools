#!/usr/bin/env bash
# Smoke-test a deployed proxy end to end.
#
#   HVYM_TOOLS_KEY=... bash scripts/smoke_proxy.sh https://img.example.com [drawing.png]
#
# Checks the things that actually break in deployment, in the order they break:
# reachability, TLS, auth, the reverse proxy's body limit, a real inference, and
# the cache. Exits non-zero on the first hard failure.
set -uo pipefail

BASE="${1:-}"
IMG="${2:-}"
KEY="${HVYM_TOOLS_KEY:-${HVYM_API_KEY:-}}"

[ -n "$BASE" ] || { echo "usage: $0 https://host [image.png]" >&2; exit 2; }
BASE="${BASE%/}"
[ -n "$KEY" ] || { echo "set HVYM_TOOLS_KEY (or HVYM_API_KEY)" >&2; exit 2; }

GRN=$(printf '\033[32m'); RED=$(printf '\033[31m'); YEL=$(printf '\033[33m'); OFF=$(printf '\033[0m')
pass=0; fail=0; warned=0
ok()   { printf '%s  PASS%s  %s\n' "$GRN" "$OFF" "$*"; pass=$((pass+1)); }
bad()  { printf '%s  FAIL%s  %s\n' "$RED" "$OFF" "$*"; fail=$((fail+1)); }
warn() { printf '%s  WARN%s  %s\n' "$YEL" "$OFF" "$*"; warned=$((warned+1)); }
hdr()  { printf '\n== %s ==\n' "$*"; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

hdr "1. Reachability and TLS"
case "$BASE" in
  https://*) ok "using https" ;;
  *) warn "not https -- the client key travels in cleartext" ;;
esac
health=$(curl -fsS -m 15 "$BASE/healthz" 2>"$TMP/err") || {
  bad "GET /healthz failed: $(tr -d '\n' < "$TMP/err")"
  echo; echo "Nothing else can pass until the proxy answers. Check:"
  echo "  - reverse proxy forwards to 127.0.0.1:8080"
  echo "  - container is up:  sudo bash install_proxy.sh --status"
  exit 1
}
ok "healthz: $health"
case "$health" in
  *'"runpod_configured":true'*) ok "RunPod configured" ;;
  *) bad "runpod_configured is false -- check the endpoint ID and key on the server" ;;
esac
case "$health" in
  *'"auth":true'*) ok "auth enabled" ;;
  *) bad "auth is DISABLED -- anyone reaching this can spend GPU time" ;;
esac

hdr "2. The door is locked"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 -X POST "$BASE/tools/reangle" \
       -F 'image=@/dev/null' 2>/dev/null)
[ "$code" = "401" ] && ok "no key -> 401" || bad "no key -> $code (expected 401)"

code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 -X POST "$BASE/tools/reangle" \
       -H "X-API-Key: definitely-not-the-key" -F 'image=@/dev/null' 2>/dev/null)
[ "$code" = "401" ] && ok "wrong key -> 401" || bad "wrong key -> $code (expected 401)"

hdr "3. Reverse-proxy body limit"
# nginx defaults client_max_body_size to 1m, which rejects real uploads long
# before the app's own limit. A 2MB body must reach the app (401, since no key)
# rather than being cut off by the web server (413).
head -c 2000000 /dev/urandom > "$TMP/2mb.bin"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 60 -X POST "$BASE/tools/reangle" \
       -F "image=@$TMP/2mb.bin" 2>/dev/null)
case "$code" in
  401) ok "2 MB body reaches the app (client_max_body_size is raised)" ;;
  413) bad "2 MB body rejected with 413 -- raise client_max_body_size to 8m and reload" ;;
  *)   warn "2 MB body -> $code (expected 401)" ;;
esac

if [ -z "$IMG" ] || [ ! -f "$IMG" ]; then
  hdr "4. Inference"
  warn "no drawing given; skipping the real request. Pass one as arg 2."
else
  hdr "4. Real inference (cold start can take ~4 min)"
  start=$(date +%s)
  code=$(curl -s -m 900 -D "$TMP/h1" -o "$TMP/out.glb" -w '%{http_code}' \
         -X POST "$BASE/tools/reangle" -H "X-API-Key: $KEY" \
         -F "image=@$IMG" -F mc_resolution=256 2>/dev/null)
  wall=$(( $(date +%s) - start ))
  if [ "$code" = "200" ]; then
    size=$(wc -c < "$TMP/out.glb")
    magic=$(head -c 4 "$TMP/out.glb")
    if [ "$magic" = "glTF" ]; then
      ok "200 in ${wall}s, ${size} bytes, valid glTF magic"
    else
      bad "200 but the body is not a .glb (magic='$magic')"
    fi
    grep -i '^x-cache\|^x-upstream-elapsed\|^x-tool-version' "$TMP/h1" | tr -d '\r' | sed 's/^/         /'
  else
    bad "expected 200, got $code after ${wall}s"
    head -c 400 "$TMP/out.glb"; echo
    [ "$code" = "504" ] && echo "         504 often means the reverse proxy's read timeout is < the cold start; set 300s+"
  fi

  hdr "5. Cache"
  code=$(curl -s -m 900 -D "$TMP/h2" -o "$TMP/out2.glb" -w '%{http_code}' \
         -X POST "$BASE/tools/reangle" -H "X-API-Key: $KEY" \
         -F "image=@$IMG" -F mc_resolution=256 2>/dev/null)
  if [ "$code" = "200" ]; then
    cache=$(grep -i '^x-cache' "$TMP/h2" | tr -d '\r' | awk '{print $2}')
    if [ "$cache" = "HIT" ]; then
      ok "repeat request served from cache (X-Cache: HIT)"
    else
      warn "repeat request was $cache -- expected HIT; is the network volume mounted?"
    fi
    if cmp -s "$TMP/out.glb" "$TMP/out2.glb"; then
      ok "cached bytes are identical to the first response"
    else
      bad "cached response differs from the original"
    fi
  else
    bad "repeat request -> $code"
  fi
fi

hdr "6. No secret leakage"
leak=0
for probe in "$BASE/healthz" "$BASE/tools/nonexistent-tool"; do
  body=$(curl -s -m 20 "$probe" 2>/dev/null; curl -s -m 20 -X POST "$probe" \
         -H "X-API-Key: $KEY" -F 'image=@/dev/null' 2>/dev/null)
  case "$body" in
    *rpa_*|*api.runpod.ai*) bad "a response mentions RunPod credentials or the upstream URL"; leak=1 ;;
  esac
done
[ "$leak" -eq 0 ] && ok "no RunPod key or upstream URL in responses"

printf '\n== summary ==\n  %d passed, %d failed, %d warnings\n' "$pass" "$fail" "$warned"
[ "$fail" -eq 0 ] || exit 1
