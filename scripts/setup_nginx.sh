#!/usr/bin/env bash
# Give the hvym-img-tools proxy its own nginx vhost + TLS certificate.
#
#   curl -fsSL https://raw.githubusercontent.com/inviti8/hvym-img-tools/main/scripts/setup_nginx.sh -o setup_nginx.sh
#   less setup_nginx.sh
#   sudo bash setup_nginx.sh img.hvym.link
#
# Written for a box that is already serving something else. The safety is in how
# it edits, not in refusing to:
#
#   * it only ever CREATES a new vhost file -- existing configs are never touched
#   * it refuses to start if `nginx -t` is already failing, so it cannot be
#     blamed for a break that was there first
#   * it records which other vhosts answer, and after reloading checks they still
#     do -- if any regressed it rolls its own file back out and reloads again
#   * every reload is preceded by `nginx -t`; a failed test rolls back and never
#     reloads
#
# Options:
#   --upstream-port N   proxy target on 127.0.0.1   (default 8080)
#   --max-upload N      MB; must match HVYM_MAX_UPLOAD_MB (default 8)
#   --read-timeout N    seconds; must exceed the cold start (default 900)
#   --email ADDR        passed to certbot for expiry notices
#   --no-tls            create the vhost but skip certbot
#   --force             overwrite an existing vhost file of the same name
#   --rollback          remove the vhost this script created, and reload
#   --retune            change only the timeout/upload limits on an existing
#                       vhost, leaving its TLS block and everything else alone
set -uo pipefail

DOMAIN=""
UPSTREAM_PORT=8080
MAX_UPLOAD=8
READ_TIMEOUT=900
EMAIL=""
DO_TLS=1
FORCE=0
ROLLBACK=0
RETUNE=0

GRN=$(printf '\033[32m'); YEL=$(printf '\033[33m'); RED=$(printf '\033[31m')
DIM=$(printf '\033[2m');  OFF=$(printf '\033[0m')
ok()   { printf '%s  ok%s   %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s warn%s  %s\n' "$YEL" "$OFF" "$*"; }
die()  { printf '%s fail%s  %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }
step() { printf '\n%s==>%s %s\n' "$GRN" "$OFF" "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --upstream-port) UPSTREAM_PORT="$2"; shift 2 ;;
    --max-upload)    MAX_UPLOAD="$2";    shift 2 ;;
    --read-timeout)  READ_TIMEOUT="$2";  shift 2 ;;
    --email)         EMAIL="$2";         shift 2 ;;
    --no-tls)        DO_TLS=0;           shift ;;
    --force)         FORCE=1;            shift ;;
    --rollback)      ROLLBACK=1;         shift ;;
    --retune)        RETUNE=1;           shift ;;
    -h|--help)       sed -n '2,30p' "$0"; exit 0 ;;
    -*)              die "unknown option: $1" ;;
    *)               DOMAIN="$1";        shift ;;
  esac
done

[ -n "$DOMAIN" ] || die "usage: sudo bash $0 <domain> [options]   (try --help)"
[ "$(id -u)" -eq 0 ] || die "run as root (sudo bash $0 $DOMAIN)"
command -v nginx >/dev/null 2>&1 || die "nginx is not installed"

# ------------------------------------------------------------ layout detection
if [ -d /etc/nginx/sites-available ] && [ -d /etc/nginx/sites-enabled ]; then
  LAYOUT="debian"
  VHOST="/etc/nginx/sites-available/$DOMAIN"
  LINK="/etc/nginx/sites-enabled/$DOMAIN"
elif [ -d /etc/nginx/conf.d ]; then
  LAYOUT="confd"
  VHOST="/etc/nginx/conf.d/${DOMAIN}.conf"
  LINK=""
else
  die "cannot find /etc/nginx/sites-available or /etc/nginx/conf.d"
fi

MARKER="# managed-by: hvym-img-tools setup_nginx.sh"

remove_vhost() {
  [ -n "$LINK" ] && rm -f "$LINK"
  rm -f "$VHOST"
}

# Undo whatever this run did to the vhost. For a fresh install that means
# deleting it; for --retune it means putting the previous file back, because
# deleting a vhost the box is already serving would turn a bad timeout into an
# outage -- and that file carries the TLS block certbot wrote.
PREV_VHOST=""
revert_vhost() {
  if [ "$RETUNE" -eq 1 ] && [ -n "$PREV_VHOST" ] && [ -f "$PREV_VHOST" ]; then
    cp -p "$PREV_VHOST" "$VHOST"
  else
    remove_vhost
  fi
}

# ------------------------------------------------------------------- rollback
if [ "$ROLLBACK" -eq 1 ]; then
  if [ ! -f "$VHOST" ]; then
    die "no vhost at $VHOST"
  fi
  grep -q "$MARKER" "$VHOST" || die "$VHOST was not created by this script -- refusing to remove it"
  remove_vhost
  if nginx -t >/dev/null 2>&1; then
    systemctl reload nginx 2>/dev/null || nginx -s reload
    ok "removed the vhost and reloaded nginx"
  else
    die "removed the vhost but 'nginx -t' still fails -- inspect manually: nginx -t"
  fi
  exit 0
fi

# ------------------------------------------------------------------ preflight
step "Preflight"

# Start from a known-good config, so a later failure is unambiguously ours.
if ! nginx -t >/dev/null 2>&1; then
  echo
  nginx -t 2>&1 | sed 's/^/    /'
  die "'nginx -t' is ALREADY failing before this script changed anything.
Fix that first -- otherwise a rollback cannot restore a working state."
fi
ok "existing nginx config tests clean ($LAYOUT layout)"

if [ "$RETUNE" -eq 1 ]; then
  [ -f "$VHOST" ] || die "--retune needs an existing vhost at $VHOST -- run without it first"
  grep -q "$MARKER" "$VHOST" || die "$VHOST was not created by this script -- refusing to edit it"
  ok "retuning our own vhost at $VHOST"
elif [ -f "$VHOST" ] && [ "$FORCE" -eq 0 ]; then
  if grep -q "$MARKER" "$VHOST"; then
    ok "re-running over our own vhost at $VHOST"
  else
    die "$VHOST already exists and was not created by this script.
Refusing to overwrite someone else's config. Pass --force if you are certain."
  fi
fi

# The proxy must actually be listening, or the vhost 502s and it looks like nginx.
# Probed with curl rather than ss: minimal images often lack iproute2.
if curl -s -o /dev/null -m 5 "http://127.0.0.1:${UPSTREAM_PORT}/healthz" 2>/dev/null; then
  ok "proxy is answering on 127.0.0.1:${UPSTREAM_PORT}"
else
  warn "nothing is answering on 127.0.0.1:${UPSTREAM_PORT}"
  warn "the vhost will return 502 until you run: sudo bash install_proxy.sh"
fi

# DNS must already point here, or certbot's HTTP-01 challenge cannot succeed.
# Irrelevant when retuning: no certificate is issued, so nothing depends on it.
[ "$RETUNE" -eq 1 ] && DO_TLS=0
resolved=$(getent hosts "$DOMAIN" 2>/dev/null | awk '{print $1}' | head -1)
if [ -z "$resolved" ]; then
  if [ "$DO_TLS" -eq 1 ]; then
    die "$DOMAIN does not resolve yet.
Add a DNS A record pointing at this box, wait for it, then re-run.
(Or pass --no-tls to create the vhost now and get the cert later.)"
  fi
  warn "$DOMAIN does not resolve yet"
else
  ok "$DOMAIN resolves to $resolved"
  mine=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^$')
  if [ -n "$mine" ] && ! printf '%s\n' "$mine" | grep -qx "$resolved"; then
    pub=$(curl -fsS -m 5 https://api.ipify.org 2>/dev/null || true)
    if [ -n "$pub" ] && [ "$pub" = "$resolved" ]; then
      ok "matches this box's public IP ($pub)"
    else
      warn "$resolved is not an address on this box${pub:+ (public IP looks like $pub)}"
      warn "if that is wrong, certbot will fail the HTTP-01 challenge"
    fi
  fi
fi

# --------------------------------------------------- snapshot the neighbours
# curl -w always emits a code (000 on failure), so no `|| echo` fallback -- that
# concatenates into "000000", which silently defeated this whole check once.
probe() {
  local h="$1" c
  c=$(curl -s -o /dev/null -w '%{http_code}' -m 10 -k "https://$h/" 2>/dev/null)
  if [ -z "$c" ] || [ "$c" = "000" ]; then
    c=$(curl -s -o /dev/null -w '%{http_code}' -m 10 "http://$h/" 2>/dev/null)
  fi
  [ -n "$c" ] || c="000"
  printf '%s' "$c"
}

step "Backing up the existing config"
BACKUP="/var/backups/nginx-hvym-$(date +%Y%m%d-%H%M%S).tar.gz"
mkdir -p /var/backups
if tar czf "$BACKUP" -C /etc nginx 2>/dev/null; then
  ok "full /etc/nginx backup: $BACKUP"
  echo "    ${DIM}restore with: sudo tar xzf $BACKUP -C /etc && sudo nginx -t && sudo systemctl reload nginx${OFF}"
else
  warn "could not write a backup to $BACKUP -- continuing, but you have no snapshot"
  BACKUP=""
fi

# Fingerprint every existing config file. Nothing outside our own new vhost may
# differ afterwards; that is checked and enforced below, not merely intended.
FINGERPRINT_BEFORE=$(mktemp)
fingerprint() {
  find /etc/nginx -type f 2>/dev/null | grep -v "^${VHOST}$" | sort \
    | xargs sha256sum 2>/dev/null
}
fingerprint > "$FINGERPRINT_BEFORE"
ok "fingerprinted $(wc -l < "$FINGERPRINT_BEFORE") existing config files"

assert_neighbours_untouched() { # $1 = context label
  local now diff
  now=$(mktemp)
  fingerprint > "$now"
  diff=$(diff <(cat "$FINGERPRINT_BEFORE") <(cat "$now") 2>/dev/null \
         | grep '^[<>]' | awk '{print $NF}' | sort -u | grep -v "^${LINK}$" || true)
  rm -f "$now"
  if [ -n "$diff" ]; then
    printf '%s  FAIL%s  %s modified files it should not have:\n' "$RED" "$OFF" "$1"
    printf '%s\n' "$diff" | sed 's/^/           /'
    return 1
  fi
  ok "$1: no existing config file was modified"
  return 0
}

step "Recording the other sites"
NEIGHBOURS=$(nginx -T 2>/dev/null \
  | awk '/^[[:space:]]*server_name/ {for(i=2;i<=NF;i++) print $i}' \
  | tr -d ';' | grep -vE '^(_|localhost|\*|)$' | grep -v "^${DOMAIN}$" \
  | sort -u | head -10)

SNAP=$(mktemp)
if [ -n "$NEIGHBOURS" ]; then
  for h in $NEIGHBOURS; do
    c=$(probe "$h")
    echo "$h $c" >> "$SNAP"
    ok "$h -> HTTP $c ${DIM}(will re-check after reload)${OFF}"
  done
else
  warn "no other vhosts detected"
fi

# ------------------------------------------------------------- write the vhost
if [ "$RETUNE" -eq 1 ]; then
  step "Retuning $VHOST"
  PREV_VHOST=$(mktemp); cp -p "$VHOST" "$PREV_VHOST"
  # Touch only the three directives this script owns. A rewrite would drop the
  # `listen 443` block, ssl_certificate lines and redirect that certbot added.
  sed -i -E     -e "s|^([[:space:]]*)client_max_body_size[[:space:]]+[0-9]+m;|\1client_max_body_size ${MAX_UPLOAD}m;|"     -e "s|^([[:space:]]*)proxy_read_timeout[[:space:]]+[0-9]+s;|\1proxy_read_timeout ${READ_TIMEOUT}s;|"     -e "s|^([[:space:]]*)proxy_send_timeout[[:space:]]+[0-9]+s;|\1proxy_send_timeout ${READ_TIMEOUT}s;|"     "$VHOST"
  if ! grep -q "proxy_read_timeout ${READ_TIMEOUT}s;" "$VHOST"; then
    cp -p "$PREV_VHOST" "$VHOST"
    die "could not find a proxy_read_timeout directive to change in $VHOST"
  fi
  ok "client_max_body_size ${MAX_UPLOAD}m, proxy_read/send_timeout ${READ_TIMEOUT}s"
  grep -c 'ssl_certificate ' "$VHOST" >/dev/null 2>&1 &&     ok "TLS block left as certbot wrote it ($(grep -c 'ssl_certificate' "$VHOST") ssl lines intact)"
else
step "Writing $VHOST"
umask 022
cat > "$VHOST" <<EOF
$MARKER
# hvym-img-tools authenticating proxy. Safe to remove with:
#     sudo bash setup_nginx.sh $DOMAIN --rollback
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:${UPSTREAM_PORT};
        proxy_set_header Host              \$host;
        proxy_set_header X-Real-IP         \$remote_addr;
        proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Both of these override nginx defaults that silently break this service:
        #   client_max_body_size defaults to 1m  -> every upload fails with 413
        #   proxy_read_timeout   defaults to 60s -> kills every cold start
        # Measured cold starts: reangle ~260s, mesh ~542s (TRELLIS pulls a
        # ~6.5GB image). 300s looked generous until mesh shipped and every cold
        # request came back as a 504 HTML page after five minutes of waiting.
        client_max_body_size ${MAX_UPLOAD}m;
        proxy_read_timeout ${READ_TIMEOUT}s;
        proxy_send_timeout ${READ_TIMEOUT}s;

        # A .glb streams straight through; buffering it just adds latency.
        proxy_buffering off;
        proxy_request_buffering off;
    }
}
EOF
ok "vhost written"

if [ "$LAYOUT" = "debian" ] && [ ! -e "$LINK" ]; then
  ln -s "$VHOST" "$LINK"
  ok "enabled via $LINK"
fi
fi

# --------------------------------------------------------------- test + reload
step "Testing config"
if ! nginx -t >/dev/null 2>&1; then
  echo; nginx -t 2>&1 | sed 's/^/    /'
  revert_vhost
  die "config test FAILED -- rolled the new vhost back out, nginx was NOT reloaded.
Your existing sites are untouched."
fi
ok "nginx -t passes"

systemctl reload nginx 2>/dev/null || nginx -s reload || {
  revert_vhost
  die "reload failed -- rolled back"
}
ok "nginx reloaded"

# nginx swaps workers asynchronously; probing instantly can still hit the old
# config and produce a confusing false alarm.
sleep 2

# ------------------------------------------------ confirm nothing else regressed
step "Confirming the other sites are unharmed"

if ! assert_neighbours_untouched "after writing the vhost"; then
  revert_vhost
  nginx -t >/dev/null 2>&1 && { systemctl reload nginx 2>/dev/null || nginx -s reload; }
  die "rolled back. Restore the backup if anything looks wrong:
    sudo tar xzf $BACKUP -C /etc && sudo nginx -t && sudo systemctl reload nginx"
fi

if [ -s "$SNAP" ]; then
  regressed=""
  while read -r h before; do
    after=$(probe "$h")
    if [ "$before" != "000" ] && { [ "$after" = "000" ] || [ "$after" -ge 502 ] 2>/dev/null; }; then
      printf '%s  FAIL%s  %s: %s -> %s\n' "$RED" "$OFF" "$h" "$before" "$after"
      regressed="$regressed $h"
    else
      ok "$h still HTTP $after ${DIM}(was $before)${OFF}"
    fi
  done < "$SNAP"

  if [ -n "$regressed" ]; then
    revert_vhost
    nginx -t >/dev/null 2>&1 && { systemctl reload nginx 2>/dev/null || nginx -s reload; }
    die "these sites regressed:$regressed
Rolled the new vhost back out and reloaded. If they are still wrong, restore:
    sudo tar xzf $BACKUP -C /etc && sudo nginx -t && sudo systemctl reload nginx"
  fi
fi
rm -f "$SNAP"

# ----------------------------------------------------------------------- certbot
install_certbot() {
  # certbot's nginx plugin must match the certbot install, so never mix
  # package managers -- pick one source and use it for both.
  if command -v apt-get >/dev/null 2>&1; then
    ok "installing certbot via apt"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq certbot python3-certbot-nginx >/dev/null 2>&1
  elif command -v dnf >/dev/null 2>&1; then
    ok "installing certbot via dnf"
    dnf install -y -q certbot python3-certbot-nginx >/dev/null 2>&1
  elif command -v yum >/dev/null 2>&1; then
    ok "installing certbot via yum"
    yum install -y -q certbot python3-certbot-nginx >/dev/null 2>&1
  elif command -v snap >/dev/null 2>&1; then
    ok "installing certbot via snap"
    snap install --classic certbot >/dev/null 2>&1
    ln -sf /snap/bin/certbot /usr/bin/certbot
  else
    return 1
  fi
  command -v certbot >/dev/null 2>&1
}

cert_exists() {
  [ -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]
}

if [ "$DO_TLS" -eq 1 ]; then
  step "TLS"

  if ! command -v certbot >/dev/null 2>&1; then
    if ! install_certbot; then
      warn "could not install certbot automatically -- the vhost is live on port 80 only"
      echo "    Debian/Ubuntu:  sudo apt install certbot python3-certbot-nginx"
      echo "    RHEL/Fedora:    sudo dnf install certbot python3-certbot-nginx"
      echo "    then:           sudo certbot --nginx -d $DOMAIN"
      DO_TLS=0
    fi
  fi

  if [ "$DO_TLS" -eq 1 ]; then
    ok "certbot $(certbot --version 2>&1 | awk '{print $2}')"

    # HTTP-01 needs port 80 reachable from the internet. Checking locally cannot
    # prove that, but an obviously-closed port is worth catching before certbot
    # burns a rate-limited failure on it.
    if command -v ss >/dev/null 2>&1 && ! ss -ltn 2>/dev/null | grep -qE ':80[[:space:]]'; then
      warn "nothing is listening on port 80 -- the HTTP-01 challenge will fail"
    fi

    if cert_exists; then
      ok "a certificate for $DOMAIN already exists -- reusing it"
      certbot --nginx -d "$DOMAIN" --non-interactive --keep-until-expiring --redirect \
        >/dev/null 2>&1 && ok "vhost wired to the existing certificate" \
        || warn "could not re-attach the existing certificate; check: certbot certificates"
    else
      set -- --nginx -d "$DOMAIN" --non-interactive --agree-tos --redirect
      if [ -n "$EMAIL" ]; then
        set -- "$@" -m "$EMAIL"
      else
        set -- "$@" --register-unsafely-without-email
        warn "no --email given; you will not get expiry warnings from Let's Encrypt"
      fi

      if certbot "$@" >/tmp/certbot.out 2>&1; then
        ok "certificate issued and HTTPS redirect configured"
      else
        sed 's/^/    /' /tmp/certbot.out | tail -15
        warn "certbot failed -- the vhost is still serving on port 80"
        echo "    Usual causes, in order of likelihood:"
        echo "      - DNS for $DOMAIN does not point at this box yet"
        echo "      - port 80 is closed in a firewall or cloud security group"
        echo "      - Let's Encrypt rate limit from earlier failed attempts"
        echo "    Retry with:  sudo certbot --nginx -d $DOMAIN"
        DO_TLS=0
      fi
    fi
  fi

  if [ "$DO_TLS" -eq 1 ] && cert_exists; then
    expiry=$(openssl x509 -enddate -noout -in "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" 2>/dev/null | cut -d= -f2)
    [ -n "$expiry" ] && ok "certificate valid until $expiry"

    # Renewal is a timer or a cron drop-in depending on how certbot was installed.
    if systemctl list-timers 2>/dev/null | grep -q certbot; then
      ok "auto-renewal active (systemd timer)"
    elif [ -f /etc/cron.d/certbot ]; then
      ok "auto-renewal active (cron)"
    else
      warn "no renewal timer found -- certs expire in 90 days"
      echo "    check with:  sudo certbot renew --dry-run"
    fi
  fi
fi

# certbot --nginx rewrites the server block it issues for and reloads nginx. It
# should only ever touch ours -- verify that rather than assume it.
if [ "$DO_TLS" -eq 1 ] && cert_exists; then
  step "Confirming certbot touched only our vhost"
  if ! assert_neighbours_untouched "certbot"; then
    warn "certbot modified a config file outside our vhost."
    warn "Nothing was rolled back automatically -- a certificate is already issued."
    echo "    Inspect the diff, and if the other site is broken, restore with:"
    echo "      sudo tar xzf $BACKUP -C /etc && sudo nginx -t && sudo systemctl reload nginx"
  fi
  if [ -s "$SNAP" ] 2>/dev/null; then :; fi
  for h in $NEIGHBOURS; do
    after=$(probe "$h")
    if [ "$after" = "000" ] || [ "$after" -ge 502 ] 2>/dev/null; then
      printf '%s  FAIL%s  %s is failing after certbot (HTTP %s)\n' "$RED" "$OFF" "$h" "$after"
      echo "    Restore immediately:"
      echo "      sudo tar xzf $BACKUP -C /etc && sudo nginx -t && sudo systemctl reload nginx"
    else
      ok "$h still HTTP $after after certbot"
    fi
  done
fi

rm -f "$FINGERPRINT_BEFORE"

# ------------------------------------------------------------------- final check
step "Verify"
scheme="https"; [ "$DO_TLS" -eq 1 ] || scheme="http"
health=$(curl -fsS -m 20 "$scheme://$DOMAIN/healthz" 2>/dev/null || true)
if [ -n "$health" ]; then
  ok "$scheme://$DOMAIN/healthz -> $health"
else
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 "$scheme://$DOMAIN/healthz" 2>/dev/null || echo 000)
  case "$code" in
    502|503) warn "$code from the vhost -- nginx is fine, the proxy container is not up.
         run: sudo bash install_proxy.sh --status" ;;
    000)     warn "no response yet -- DNS may still be propagating" ;;
    *)       warn "unexpected $code from $scheme://$DOMAIN/healthz" ;;
  esac
fi

step "Done"
cat <<EOF
  client URL:  $scheme://$DOMAIN/tools/reangle
  smoke test:  HVYM_TOOLS_KEY=<key> bash scripts/smoke_proxy.sh $scheme://$DOMAIN drawing.png
  undo:        sudo bash $0 $DOMAIN --rollback
EOF
