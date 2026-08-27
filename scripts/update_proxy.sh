#!/usr/bin/env bash
# Update an installed hvym-img-tools proxy to a new image, safely.
#
#   curl -fsSL https://raw.githubusercontent.com/inviti8/hvym-img-tools/main/scripts/update_proxy.sh -o update_proxy.sh
#   sudo bash update_proxy.sh                 # latest pinned tag below
#   sudo bash update_proxy.sh 0.1.2           # a specific tag
#   sudo bash update_proxy.sh --rollback      # back to the previous image
#   sudo bash update_proxy.sh --status
#
# Safety, in the order it matters on a box that also serves something else:
#
#   * the new image is PULLED FIRST -- a failed or slow pull never takes the
#     running proxy down, because nothing is stopped until the bytes are local
#   * the currently-running image is recorded before the swap, so --rollback is
#     a real option rather than "hope the old tag still exists"
#   * after the swap the proxy must answer /healthz AND reject an unauthenticated
#     request; if it does not, this script puts the old image back by itself
#   * nginx is never touched -- the container keeps the same name and port, so
#     the vhost pointing at 127.0.0.1:8080 needs no change
#
# Existing configuration in /etc/hvym-img-tools/proxy.env is reused untouched:
# your keys and endpoint id survive the update, and are never re-prompted.
set -uo pipefail

DEFAULT_TAG="0.1.2"
REGISTRY="ghcr.io/inviti8/hvym-img-proxy"
NAME="hvym-img-proxy"
CONF_DIR="/etc/hvym-img-tools"
ENV_FILE="$CONF_DIR/proxy.env"
PREV_FILE="$CONF_DIR/previous-image"
PORT="${HVYM_BIND_PORT:-8080}"
BIND="${HVYM_BIND_ADDR:-127.0.0.1}"
MEM_LIMIT="${HVYM_MEM_LIMIT:-512m}"
MEM_RESERVE="${HVYM_MEM_RESERVE:-256m}"
CPUS="${HVYM_CPUS:-0.5}"

GRN=$(printf '\033[32m'); YEL=$(printf '\033[33m'); RED=$(printf '\033[31m')
DIM=$(printf '\033[2m');  OFF=$(printf '\033[0m')
ok()   { printf '%s  ok%s   %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s warn%s  %s\n' "$YEL" "$OFF" "$*"; }
die()  { printf '%s fail%s  %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }
step() { printf '\n%s==>%s %s\n' "$GRN" "$OFF" "$*"; }

ACTION="update"
TAG="$DEFAULT_TAG"
while [ $# -gt 0 ]; do
  case "$1" in
    --rollback) ACTION="rollback"; shift ;;
    --status)   ACTION="status";   shift ;;
    -h|--help)  sed -n '2,22p' "$0"; exit 0 ;;
    -*)         die "unknown option: $1" ;;
    *)          TAG="$1"; shift ;;
  esac
done

require_docker() {
  command -v docker >/dev/null 2>&1 || die "docker is not installed"
  docker info >/dev/null 2>&1 || die "the docker daemon is not responding"
}

running_image() {
  docker inspect -f '{{.Config.Image}}' "$NAME" 2>/dev/null || true
}

# ------------------------------------------------------------------ status
if [ "$ACTION" = "status" ]; then
  require_docker
  cur=$(running_image)
  if [ -z "$cur" ]; then
    echo "not running"
    exit 1
  fi
  echo "  image:    $cur"
  [ -f "$PREV_FILE" ] && echo "  previous: $(cat "$PREV_FILE")"
  docker ps -f "name=^${NAME}$" --format '  status:   {{.Status}}'
  docker stats --no-stream --format '  usage:    mem {{.MemUsage}}  cpu {{.CPUPerc}}' "$NAME"
  echo "  health:   $(curl -fsS -m 5 "http://${BIND}:${PORT}/healthz" 2>/dev/null || echo unreachable)"
  echo "  warm:     $(curl -fsS -m 10 "http://${BIND}:${PORT}/warm" 2>/dev/null || echo 'no /warm endpoint (older image)')"
  exit 0
fi

[ "$(id -u)" -eq 0 ] || die "run as root (sudo bash $0 $*)"
require_docker
[ -f "$ENV_FILE" ] || die "$ENV_FILE not found -- run install_proxy.sh first"

# ---------------------------------------------------------------- rollback
if [ "$ACTION" = "rollback" ]; then
  [ -f "$PREV_FILE" ] || die "no previous image recorded; nothing to roll back to"
  TARGET=$(cat "$PREV_FILE")
  step "Rolling back to $TARGET"
else
  TARGET="${REGISTRY}:${TAG}"
fi

CURRENT=$(running_image)
[ -n "$CURRENT" ] && ok "currently running: $CURRENT" || warn "no container running (this will start one)"

if [ "$CURRENT" = "$TARGET" ] && [ "$ACTION" != "rollback" ]; then
  ok "already on $TARGET -- nothing to do"
  echo "    (re-run with a different tag, or --rollback, if you meant something else)"
  exit 0
fi

# --------------------------------------------------------------------- pull
# Pull BEFORE stopping anything: a slow or failed pull must never be able to
# leave the box with no proxy running.
step "Pulling $TARGET"
if ! docker pull "$TARGET" >/dev/null 2>&1; then
  die "could not pull $TARGET -- the running proxy was NOT touched.
Check the tag exists: https://github.com/inviti8/hvym-img-tools/pkgs/container/hvym-img-proxy"
fi
ok "pulled (the running proxy is still untouched)"

# --------------------------------------------------------------------- swap
step "Swapping the container"
if [ -n "$CURRENT" ]; then
  printf '%s\n' "$CURRENT" > "$PREV_FILE"
  chmod 600 "$PREV_FILE"
  docker rm -f "$NAME" >/dev/null 2>&1 && ok "stopped the old container"
fi

start_container() {
  docker run -d --name "$NAME" \
    --restart=unless-stopped \
    -p "${BIND}:${PORT}:8080" \
    --memory="$MEM_LIMIT" --memory-reservation="$MEM_RESERVE" --cpus="$CPUS" \
    --env-file "$ENV_FILE" \
    "$1" >/dev/null 2>&1
}

if ! start_container "$TARGET"; then
  warn "the new container failed to start; restoring $CURRENT"
  docker rm -f "$NAME" >/dev/null 2>&1
  start_container "$CURRENT" && die "rolled back to $CURRENT -- the update did not apply" \
    || die "could not restart the old image either. Run: sudo bash install_proxy.sh"
fi
ok "started $NAME on $TARGET"

# ------------------------------------------------------------------- verify
step "Verifying"
health=""
for _ in $(seq 1 30); do
  health=$(curl -fsS -m 3 "http://${BIND}:${PORT}/healthz" 2>/dev/null) && break
  sleep 1
done

rollback_now() { # $1 = why
  warn "$1"
  docker logs --tail 30 "$NAME" 2>&1 | sed 's/^/    /' || true
  if [ -n "$CURRENT" ]; then
    docker rm -f "$NAME" >/dev/null 2>&1
    if start_container "$CURRENT"; then
      die "AUTOMATICALLY ROLLED BACK to $CURRENT. The update did not apply."
    fi
  fi
  die "update failed and no previous image was available. Run: sudo bash install_proxy.sh"
}

[ -n "$health" ] || rollback_now "the new image never became healthy"
ok "healthz: $health"

case "$health" in
  *'"auth":true'*) ok "auth is enabled" ;;
  *) rollback_now "auth is DISABLED on the new image -- refusing to leave it running" ;;
esac
case "$health" in
  *'"runpod_configured":true'*) ok "RunPod is configured" ;;
  *) warn "runpod_configured is false -- check $ENV_FILE" ;;
esac

# Prove the door is still locked, rather than trusting the flag.
code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 \
       -X POST "http://${BIND}:${PORT}/tools/reangle" -F 'image=@/dev/null' 2>/dev/null)
[ "$code" = "401" ] || rollback_now "an unauthenticated request returned $code, expected 401"
ok "unauthenticated request rejected (401)"

# /warm is new in 0.1.1+; absent on older images, which is not a failure.
warm=$(curl -fsS -m 15 "http://${BIND}:${PORT}/warm" 2>/dev/null || true)
if [ -n "$warm" ]; then
  ok "warm lease endpoint live: $warm"
else
  warn "no /warm endpoint on this image (expected on 0.1.0 and earlier)"
fi

step "Done"
cat <<EOF
  updated:  ${CURRENT:-none} -> $TARGET
  status:   sudo bash $0 --status
  rollback: sudo bash $0 --rollback
  logs:     docker logs -f $NAME

  nginx was not touched: same container name, same ${BIND}:${PORT}.
EOF
