#!/usr/bin/env bash
# Point a tool at its own RunPod serverless endpoint, on an installed proxy.
#
#   curl -fsSLO https://raw.githubusercontent.com/inviti8/hvym-img-tools/main/scripts/set_tool_endpoint.sh
#   sudo bash set_tool_endpoint.sh mesh km99b7mrj2f85r
#   sudo bash set_tool_endpoint.sh --list
#   sudo bash set_tool_endpoint.sh --remove mesh
#
# Tools live on separate endpoints (docs/tools/mesh.md), so the proxy resolves
# RUNPOD_ENDPOINT_ID_<TOOL> first and falls back to RUNPOD_ENDPOINT_ID. This
# edits only that one variable in /etc/hvym-img-tools/proxy.env and RECREATES the
# container; it never touches your keys, and never touches nginx.
#
# Recreate, not restart: docker reads --env-file once at `run` and bakes the
# result into the container config, so `docker restart` silently ignores any
# edit to that file.
#
# Separate from update_proxy.sh on purpose: that script changes which IMAGE
# runs, this changes CONFIGURATION. Conflating them would mean you could not fix
# a wrong endpoint id without also pulling a new image.
set -uo pipefail

NAME="hvym-img-proxy"
CONF_DIR="/etc/hvym-img-tools"
ENV_FILE="$CONF_DIR/proxy.env"
PORT="${HVYM_BIND_PORT:-8080}"
BIND="${HVYM_BIND_ADDR:-127.0.0.1}"

GRN=$(printf '\033[32m'); YEL=$(printf '\033[33m'); RED=$(printf '\033[31m')
DIM=$(printf '\033[2m');  OFF=$(printf '\033[0m')
ok()   { printf '%s  ok%s   %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s warn%s  %s\n' "$YEL" "$OFF" "$*"; }
die()  { printf '%s fail%s  %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }
step() { printf '\n%s==>%s %s\n' "$GRN" "$OFF" "$*"; }

ACTION="set"
TOOL=""
EPID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --list)    ACTION="list";   shift ;;
    --remove)  ACTION="remove"; TOOL="${2:-}"; shift 2 || shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    -*)        die "unknown option: $1" ;;
    *)         if [ -z "$TOOL" ]; then TOOL="$1"; else EPID="$1"; fi; shift ;;
  esac
done

command -v docker >/dev/null 2>&1 || die "docker is not installed"

show_config() {
  [ -f "$ENV_FILE" ] || die "$ENV_FILE not found -- run install_proxy.sh first"
  echo "  default : $(grep -E '^RUNPOD_ENDPOINT_ID=' "$ENV_FILE" | cut -d= -f2- || echo '(none)')"
  local any=0
  while IFS= read -r line; do
    any=1
    local t="${line%%=*}"; t="${t#RUNPOD_ENDPOINT_ID_}"
    printf '  %-8s: %s\n' "$(echo "$t" | tr 'A-Z' 'a-z')" "${line#*=}"
  done < <(grep -E '^RUNPOD_ENDPOINT_ID_' "$ENV_FILE" 2>/dev/null || true)
  [ "$any" -eq 1 ] || echo "  (no per-tool endpoints; everything uses the default)"
}

if [ "$ACTION" = "list" ]; then
  step "Configured endpoints"
  show_config
  echo
  step "What the running proxy reports"
  curl -fsS -m 5 "http://${BIND}:${PORT}/healthz" 2>/dev/null || echo "  (proxy unreachable)"
  echo
  exit 0
fi

[ "$(id -u)" -eq 0 ] || die "run as root (sudo bash $0 $*)"
[ -f "$ENV_FILE" ] || die "$ENV_FILE not found -- run install_proxy.sh first"
[ -n "$TOOL" ] || die "usage: $0 <tool> <endpoint-id>   (or --list / --remove <tool>)"

KEY="RUNPOD_ENDPOINT_ID_$(echo "$TOOL" | tr 'a-z-' 'A-Z_')"

step "Before"
show_config

# Back up the whole env file: it holds the RunPod key, and a botched edit here
# is far more annoying to recover than a stale endpoint id.
BACKUP="${ENV_FILE}.$(date +%Y%m%d-%H%M%S).bak"
cp -p "$ENV_FILE" "$BACKUP"
chmod 600 "$BACKUP"
ok "backed up to $BACKUP"

# Rewrite through a temp file so a failure cannot leave a half-written env.
TMP=$(mktemp); chmod 600 "$TMP"
grep -vE "^${KEY}=" "$ENV_FILE" > "$TMP"

if [ "$ACTION" = "remove" ]; then
  step "Removing $KEY"
else
  [ -n "$EPID" ] || die "no endpoint id given: $0 $TOOL <endpoint-id>"
  case "$EPID" in
    *[!a-zA-Z0-9]*) die "endpoint id looks wrong: '$EPID' (expected alphanumeric)" ;;
  esac
  step "Setting $KEY=$EPID"
  printf '%s=%s\n' "$KEY" "$EPID" >> "$TMP"
fi

mv "$TMP" "$ENV_FILE"
chmod 600 "$ENV_FILE"

step "Recreating the proxy"
# NOT `docker restart`. --env-file is read once, at `docker run`, and baked into
# the container's config; a restart reuses the ORIGINAL environment and silently
# ignores every edit made here. That cost a debugging round: the variable was
# written correctly, the container never saw it, and this script's own warning
# blamed the image version.
IMAGE=$(docker inspect -f '{{.Config.Image}}' "$NAME" 2>/dev/null)
[ -n "$IMAGE" ] || die "$NAME is not running; start it with install_proxy.sh first"
ok "reusing image $IMAGE"

start_proxy() {
  docker run -d --name "$NAME" \
    --restart=unless-stopped \
    -p "${BIND}:${PORT}:8080" \
    --memory="${HVYM_MEM_LIMIT:-512m}" \
    --memory-reservation="${HVYM_MEM_RESERVE:-256m}" \
    --cpus="${HVYM_CPUS:-0.5}" \
    --env-file "$ENV_FILE" \
    "$IMAGE" >/dev/null 2>&1
}

docker rm -f "$NAME" >/dev/null 2>&1
if ! start_proxy; then
  cp -p "$BACKUP" "$ENV_FILE"
  start_proxy && die "could not start with the new config -- restored the previous one" \
    || die "could not restart the proxy at all. Run: sudo bash install_proxy.sh"
fi

health=""
for _ in $(seq 1 30); do
  health=$(curl -fsS -m 3 "http://${BIND}:${PORT}/healthz" 2>/dev/null) && break
  sleep 1
done

if [ -z "$health" ]; then
  warn "the proxy did not come back healthy; restoring the previous config"
  cp -p "$BACKUP" "$ENV_FILE"
  docker rm -f "$NAME" >/dev/null 2>&1
  start_proxy
  die "rolled back. Check: docker logs $NAME"
fi
ok "healthz: $health"

case "$health" in
  *'"auth":true'*) ok "auth still enabled" ;;
  *) warn "auth is DISABLED -- check HVYM_API_KEY in $ENV_FILE" ;;
esac

if [ "$ACTION" = "set" ]; then
  case "$health" in
    *"\"$TOOL\""*) ok "the proxy is now routing '$TOOL' to its own endpoint" ;;
    *) warn "'$TOOL' is not in tool_endpoints. Either the container did not pick up
         the env file, or the image predates per-tool routing (needs 0.3.0+):
           docker inspect $NAME --format '{{.Config.Image}}'
           sudo bash update_proxy.sh 0.3.1" ;;
  esac
fi

step "After"
show_config
cat <<EOF

  Test it:
    curl -X POST https://<your-domain>/tools/${TOOL} \\
         -H "X-API-Key: \$KEY" -F image=@sketch.png -o reference.glb

  ${DIM}Undo: sudo cp $BACKUP $ENV_FILE && sudo bash $0 --list${OFF}
  ${DIM}      (then re-run this script; a plain 'docker restart' would NOT
  ${DIM}       pick the restored file up)${OFF}
EOF
