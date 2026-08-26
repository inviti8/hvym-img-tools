#!/usr/bin/env bash
# Install the hvym-img-tools authenticating proxy on a Linux host.
#
#   curl -fsSL https://raw.githubusercontent.com/inviti8/hvym-img-tools/main/scripts/install_proxy.sh -o install_proxy.sh
#   less install_proxy.sh          # read it before running anything as root
#   sudo bash install_proxy.sh
#
# Safe to re-run: it upgrades in place rather than piling up containers.
#
# Designed for a box that is ALREADY running something else. It therefore:
#   * binds to 127.0.0.1 only, so nothing is exposed until you choose to
#   * caps memory, so a runaway proxy is OOM-killed instead of starving the neighbour
#   * NEVER edits your web-server config -- it prints a snippet for you to paste,
#     because auto-editing a live reverse proxy can take the other service down
#
# Actions:  (default) install/upgrade   --status   --uninstall   --print-proxy-conf
set -euo pipefail

IMAGE="ghcr.io/inviti8/hvym-img-proxy:0.1.0"
NAME="hvym-img-proxy"
PORT="${HVYM_BIND_PORT:-8080}"
BIND="${HVYM_BIND_ADDR:-127.0.0.1}"
CONF_DIR="/etc/hvym-img-tools"
ENV_FILE="$CONF_DIR/proxy.env"
MEM_LIMIT="${HVYM_MEM_LIMIT:-512m}"
MEM_RESERVE="${HVYM_MEM_RESERVE:-256m}"
CPUS="${HVYM_CPUS:-0.5}"
MAX_UPLOAD_MB="${HVYM_MAX_UPLOAD_MB:-8}"
PROXY_TIMEOUT="${HVYM_PROXY_TIMEOUT:-600}"
# 512m limit + headroom for the image pull and for the neighbour.
MIN_FREE_MB=700

GRN=$(printf '\033[32m'); YEL=$(printf '\033[33m'); RED=$(printf '\033[31m')
DIM=$(printf '\033[2m');  OFF=$(printf '\033[0m')
ok()   { printf '%s  ok%s  %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s warn%s %s\n' "$YEL" "$OFF" "$*"; }
die()  { printf '%s fail%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }
step() { printf '\n%s==>%s %s\n' "$GRN" "$OFF" "$*"; }

FOUND_WEB=""

need_root() {
  [ "$(id -u)" -eq 0 ] || die "run as root (sudo bash $0)"
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    die "docker is not installed.

This script will not install it for you: on a box already running another
service, bringing in a container runtime is a change to make deliberately.
See https://docs.docker.com/engine/install/ then re-run."
  fi
  docker info >/dev/null 2>&1 || die "docker is installed but the daemon is not responding"
}

# ---------------------------------------------------------------- preflight
preflight() {
  step "Preflight"
  require_docker
  ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo present)"

  local avail
  avail=$(free -m 2>/dev/null | awk '/^Mem:/ {print ($7 != "" ? $7 : $4)}' || true)
  if [ -n "${avail:-}" ]; then
    if [ "$avail" -lt "$MIN_FREE_MB" ]; then
      warn "only ${avail} MB available; the proxy is capped at ${MEM_LIMIT} but the pull needs room"
      warn "installing anyway -- watch the neighbour with: docker stats"
    else
      ok "${avail} MB available (proxy uses ~75 MB steady, hard cap ${MEM_LIMIT})"
    fi
  fi

  local running_already=0
  [ -n "$(docker ps -q -f "name=^${NAME}$")" ] && running_already=1

  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"; then
    if [ "$running_already" -eq 1 ]; then
      ok "port ${PORT} held by the existing ${NAME} (this is an upgrade)"
    else
      die "port ${PORT} is already in use by something else.
Set HVYM_BIND_PORT=<free port> and re-run."
    fi
  else
    ok "port ${PORT} is free"
  fi

  local svc
  for svc in nginx caddy apache2 traefik haproxy; do
    if pgrep -x "$svc" >/dev/null 2>&1; then
      ok "found $svc running -- reuse it for TLS (snippet printed at the end)"
      FOUND_WEB="$svc"
    fi
  done
  [ -n "$FOUND_WEB" ] || warn "no web server detected; you will need one for TLS"
}

# ------------------------------------------------------------------- config
gen_key() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -base64 48 | tr -dc 'A-Za-z0-9' | cut -c1-43
  else
    tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 43
  fi
}

configure() {
  step "Configuration"
  mkdir -p "$CONF_DIR"
  chmod 700 "$CONF_DIR"

  if [ -f "$ENV_FILE" ]; then
    ok "reusing $ENV_FILE (delete it to reconfigure)"
    return
  fi

  echo "  Secrets go to $ENV_FILE (0600) and are passed with --env-file,"
  echo "  so they stay out of your shell history and out of 'ps'."
  echo

  local runpod_key endpoint_id api_key
  if [ -n "${RUNPOD_API_KEY:-}" ]; then
    runpod_key="$RUNPOD_API_KEY"
    ok "RUNPOD_API_KEY taken from the environment"
  else
    printf '  RunPod API key (input hidden): '
    read -rs runpod_key < /dev/tty
    echo
  fi
  [ -n "$runpod_key" ] || die "RunPod API key is required"

  if [ -n "${RUNPOD_ENDPOINT_ID:-}" ]; then
    endpoint_id="$RUNPOD_ENDPOINT_ID"
    ok "RUNPOD_ENDPOINT_ID taken from the environment"
  else
    printf '  RunPod serverless endpoint ID: '
    read -r endpoint_id < /dev/tty
  fi
  [ -n "$endpoint_id" ] || die "endpoint ID is required"

  if [ -n "${HVYM_API_KEY:-}" ]; then
    api_key="$HVYM_API_KEY"
    ok "HVYM_API_KEY taken from the environment"
  else
    api_key="$(gen_key)"
    ok "generated a scoped client key (printed at the end)"
  fi

  umask 077
  cat > "$ENV_FILE" <<EOF
# hvym-img-tools proxy. Written by install_proxy.sh -- keep this at 0600.
# RUNPOD_API_KEY grants FULL RunPod account access. It must never leave this box.
RUNPOD_API_KEY=$runpod_key
RUNPOD_ENDPOINT_ID=$endpoint_id
# The scoped key Inkternity presents. Rotate by editing this and re-running.
HVYM_API_KEY=$api_key
HVYM_MAX_UPLOAD_MB=$MAX_UPLOAD_MB
HVYM_PROXY_TIMEOUT=$PROXY_TIMEOUT
HVYM_PORT=8080
EOF
  chmod 600 "$ENV_FILE"
  ok "wrote $ENV_FILE"
}

# ------------------------------------------------------------------ install
install_container() {
  step "Image"
  docker pull "$IMAGE" >/dev/null 2>&1 || die "could not pull $IMAGE"
  ok "pulled $IMAGE"

  step "Container"
  if [ -n "$(docker ps -aq -f "name=^${NAME}$")" ]; then
    docker rm -f "$NAME" >/dev/null
    ok "removed the previous container"
  fi

  docker run -d --name "$NAME" \
    --restart=unless-stopped \
    -p "${BIND}:${PORT}:8080" \
    --memory="$MEM_LIMIT" --memory-reservation="$MEM_RESERVE" --cpus="$CPUS" \
    --env-file "$ENV_FILE" \
    "$IMAGE" >/dev/null
  ok "started $NAME on ${BIND}:${PORT} (mem ${MEM_LIMIT}, cpus ${CPUS})"
}

verify() {
  step "Verify"
  local i health=""
  for i in $(seq 1 30); do
    if health=$(curl -fsS -m 3 "http://${BIND}:${PORT}/healthz" 2>/dev/null); then
      break
    fi
    sleep 1
  done
  if [ -z "$health" ]; then
    docker logs --tail 30 "$NAME" >&2 || true
    die "the proxy did not become healthy -- logs above"
  fi
  ok "healthz: $health"

  case "$health" in
    *'"runpod_configured":true'*) ok "RunPod credentials are configured" ;;
    *) warn "runpod_configured is false -- check RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID in $ENV_FILE" ;;
  esac

  case "$health" in
    *'"auth":true'*) ok "auth is enabled" ;;
    *) die "auth is DISABLED -- refusing to report success. Check HVYM_API_KEY in $ENV_FILE" ;;
  esac

  # Prove the door is locked rather than trusting the flag.
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 \
         -X POST "http://${BIND}:${PORT}/tools/reangle" \
         -F 'image=@/dev/null' 2>/dev/null || echo 000)
  if [ "$code" = "401" ]; then
    ok "an unauthenticated request is rejected (401)"
  else
    die "expected 401 for an unauthenticated request, got ${code} -- do not expose this"
  fi
}

print_proxy_conf() {
  local key=""
  if [ -f "$ENV_FILE" ]; then
    key=$(grep -E '^HVYM_API_KEY=' "$ENV_FILE" | cut -d= -f2- || true)
  fi

  printf '\n%s==>%s Next: put TLS in front of it\n\n' "$GRN" "$OFF"
  cat <<EOF
The proxy listens on ${BIND}:${PORT} and is NOT reachable from outside this box.
That is deliberate. Terminate TLS with the web server you already run -- over
plain HTTP the client key is cleartext on the wire.

${DIM}This script does not edit your web-server config: another service runs on this
box, and a bad edit plus a reload would take it down. Paste this instead.${OFF}

--- nginx -------------------------------------------------------------------
location /tools/ {
    proxy_pass http://${BIND}:${PORT};
    proxy_set_header Host \$host;
    proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;

    # BOTH of these matter -- nginx's defaults break this service:
    #   client_max_body_size defaults to 1m  -> every upload fails with 413
    #   proxy_read_timeout   defaults to 60s -> kills every cold start (~260s)
    client_max_body_size ${MAX_UPLOAD_MB}m;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
}
location = /healthz { proxy_pass http://${BIND}:${PORT}/healthz; }

--- Caddy -------------------------------------------------------------------
img.example.com {
    reverse_proxy ${BIND}:${PORT} {
        transport http { read_timeout 300s }
    }
    request_body { max_size ${MAX_UPLOAD_MB}MB }
}
-----------------------------------------------------------------------------

Test the config before reloading:  nginx -t && systemctl reload nginx
Then:                              curl https://your-domain/healthz
EOF

  if [ -n "$key" ]; then
    printf '\n%s==>%s The client key for Inkternity\n\n    %s\n\n' "$GRN" "$OFF" "$key"
    cat <<EOF
Give this to the client build. RUNPOD_API_KEY stays on this box only.
Rotate by editing $ENV_FILE and re-running this script.

Client contract: docs/CLIENT.md ${DIM}-- note the request timeout must be >= 300s${OFF}
EOF
  fi
}

do_status() {
  require_docker
  if [ -z "$(docker ps -q -f "name=^${NAME}$")" ]; then
    echo "not running"
    docker ps -a -f "name=^${NAME}$" --format '  last state: {{.Status}}' || true
    exit 1
  fi
  docker ps -f "name=^${NAME}$" --format '  {{.Names}}  {{.Status}}  {{.Ports}}'
  docker stats --no-stream --format '  mem {{.MemUsage}}   cpu {{.CPUPerc}}' "$NAME"
  curl -fsS -m 3 "http://${BIND}:${PORT}/healthz" && echo
}

do_uninstall() {
  need_root
  require_docker
  if docker rm -f "$NAME" >/dev/null 2>&1; then
    ok "removed container"
  else
    warn "no container to remove"
  fi
  echo "Left in place: $ENV_FILE (it holds your keys)."
  echo "Remove it yourself when you are done:  rm -rf $CONF_DIR"
}

main() {
  case "${1:-install}" in
    --status)           do_status; exit 0 ;;
    --uninstall)        do_uninstall; exit 0 ;;
    --print-proxy-conf) print_proxy_conf; exit 0 ;;
    -h|--help)          sed -n '2,17p' "$0"; exit 0 ;;
    install)            ;;
    *)                  die "unknown action: $1 (try --help)" ;;
  esac

  need_root
  echo "hvym-img-tools proxy installer"
  preflight
  configure
  install_container
  verify
  print_proxy_conf

  step "Done"
  echo "  status:    sudo bash $0 --status"
  echo "  logs:      docker logs -f $NAME"
  echo "  uninstall: sudo bash $0 --uninstall"
}

main "$@"
