set -u
PASS=0; FAIL=0
check() {
  if printf '%s' "$3" | grep -qF "$2"; then echo "  [PASS] $1"; PASS=$((PASS+1));
  else echo "  [FAIL] $1"; echo "         wanted: $2"; echo "         got: $(printf '%s' "$3" | tr '\n' '|' | head -c 350)"; FAIL=$((FAIL+1)); fi
}

mkdir -p /etc/hvym-img-tools
cat > /etc/hvym-img-tools/proxy.env <<'E'
RUNPOD_API_KEY=fake
RUNPOD_ENDPOINT_ID=ep123
HVYM_API_KEY=aaaaaaaaaaaaaaaaaaaaaaaa
E
chmod 600 /etc/hvym-img-tools/proxy.env

# --- stub docker -------------------------------------------------------------
# STATE controls what the fake daemon does, so each scenario is exact.
cat > /usr/local/bin/docker <<'D'
#!/usr/bin/env bash
echo "docker $*" >> /tmp/docker.calls
RUNNING_F=/tmp/running_image
case "$1" in
  info|version) exit 0 ;;
  pull)
    [ -f /tmp/pull_fails ] && exit 1
    exit 0 ;;
  inspect)
    case "$*" in
      *"{{.Image}}"*)
        # digest is derived from the image the container was STARTED with,
        # recorded at run time -- a moving tag cannot change it afterwards
        [ -f /tmp/running_digest ] && cat /tmp/running_digest && exit 0
        exit 1 ;;
      *"image inspect"*) exit 0 ;;
      *)
        [ -f "$RUNNING_F" ] && cat "$RUNNING_F" && exit 0
        exit 1 ;;
    esac ;;
  image) exit 0 ;;
  rm) rm -f "$RUNNING_F"; exit 0 ;;
  run)
    img="${@: -1}"
    if [ -f /tmp/start_fails ] && [ "$img" != "$(cat /tmp/prev_ok 2>/dev/null)" ]; then exit 1; fi
    echo "$img" > "$RUNNING_F"
    # :latest resolves to whatever RESOLVE says right now; a pinned tag is itself
    case "$img" in
      *:latest) cat /tmp/latest_resolves_to 2>/dev/null > /tmp/running_digest || echo "sha256:unknown" > /tmp/running_digest ;;
      sha256:*) echo "$img" > /tmp/running_digest ;;
      *) echo "sha256:$(printf '%s' "$img" | tr -c 'a-z0-9' '_')" > /tmp/running_digest ;;
    esac
    exit 0 ;;
  ps) [ -f "$RUNNING_F" ] && echo "hvym-img-proxy Up 1s" ; exit 0 ;;
  stats) echo "  mem 74MiB / 4GiB  cpu 0.3%" ; exit 0 ;;
  logs) echo "(container logs)"; exit 0 ;;
esac
exit 0
D
chmod +x /usr/local/bin/docker

# --- stub curl ---------------------------------------------------------------
cat > /usr/local/bin/curl <<'C'
#!/usr/bin/env bash
for a in "$@"; do
  case "$a" in
    *"/tools/reangle") printf '%s' "${FAKE_TOOLS_CODE:-401}"; exit 0 ;;
    *"/warm") printf '{"state":"cold","ready":false,"active_leases":0}'; exit 0 ;;
    *"/healthz")
        [ -f /tmp/unhealthy ] && exit 7
        printf '%s' "${FAKE_HEALTH:-{\"status\":\"ok\",\"auth\":true,\"runpod_configured\":true}}"
        exit 0 ;;
  esac
done
exit 0
C
chmod +x /usr/local/bin/curl

echo "=========== TEST 1: fresh update (nothing running) ==========="
out=$(bash /work/update_proxy.sh 0.1.2 2>&1)
check "pulled first"            "pulled (the running proxy is still untouched)" "$out"
check "started"                 "started hvym-img-proxy" "$out"
check "verified auth"           "auth is enabled" "$out"
check "verified 401"            "unauthenticated request rejected" "$out"
check "saw /warm"               "warm lease endpoint live" "$out"
check "running image is 0.1.2"  "0.1.2" "$(cat /tmp/running_image)"

echo
echo "=========== TEST 2: idempotent re-run ==========="
out=$(bash /work/update_proxy.sh 0.1.2 2>&1)
check "no-op when already there" "already on" "$out"

echo
echo "=========== TEST 3: upgrade records the previous image ==========="
out=$(bash /work/update_proxy.sh 0.1.3 2>&1)
check "upgraded"                "0.1.2 -> ghcr.io/inviti8/hvym-img-proxy:0.1.3" "$out"
check "previous recorded"       "0.1.2" "$(cat /etc/hvym-img-tools/previous-image)"

echo
echo "=========== TEST 4: failed pull must NOT touch the running proxy ==========="
touch /tmp/pull_fails
before=$(cat /tmp/running_image)
out=$(bash /work/update_proxy.sh 0.9.9 2>&1)
rm -f /tmp/pull_fails
check "refused"                 "could not pull" "$out"
check "said it was untouched"   "was NOT touched" "$out"
[ "$(cat /tmp/running_image)" = "$before" ] && { echo "  [PASS] running container unchanged"; PASS=$((PASS+1)); } || { echo "  [FAIL] running container changed"; FAIL=$((FAIL+1)); }

echo
echo "=========== TEST 5: unhealthy new image rolls back automatically ==========="
cat /tmp/running_image > /tmp/prev_ok
touch /tmp/unhealthy
out=$(bash /work/update_proxy.sh 0.1.4 2>&1)
rm -f /tmp/unhealthy
check "detected unhealthy"      "never became healthy" "$out"
check "rolled back"             "AUTOMATICALLY ROLLED BACK" "$out"
check "restored old image (by digest)" "0_1_3" "$(cat /tmp/running_image)"

echo
echo "=========== TEST 6: 401 check failing rolls back ==========="
cat /tmp/running_image > /tmp/prev_ok
out=$(FAKE_TOOLS_CODE=200 bash /work/update_proxy.sh 0.1.5 2>&1)
check "caught the open door"    "expected 401" "$out"
check "rolled back"             "AUTOMATICALLY ROLLED BACK" "$out"

echo
echo "=========== TEST 7: auth disabled rolls back ==========="
cat /tmp/running_image > /tmp/prev_ok
out=$(FAKE_HEALTH='{"status":"ok","auth":false,"runpod_configured":true}' bash /work/update_proxy.sh 0.1.6 2>&1)
check "refused auth-disabled"   "auth is DISABLED" "$out"
check "rolled back"             "AUTOMATICALLY ROLLED BACK" "$out"

echo
echo "=========== TEST 8: --rollback ==========="
out=$(bash /work/update_proxy.sh --rollback 2>&1)
check "rolled back on demand"   "Rolling back to" "$out"

echo
echo "=========== TEST 9: --status ==========="
out=$(bash /work/update_proxy.sh --status 2>&1)
check "reports image"           "image:" "$out"
check "reports warm"            "warm:" "$out"

echo
echo "=========== TEST 10: refuses without config ==========="
mv /etc/hvym-img-tools/proxy.env /tmp/env.bak
out=$(bash /work/update_proxy.sh 0.1.2 2>&1)
mv /tmp/env.bak /etc/hvym-img-tools/proxy.env
check "needs install first"     "run install_proxy.sh first" "$out"

echo
echo "=========== TEST 11: :latest rollback goes back to the OLD build ==========="
# This is the trap the digest exists for: deploy :latest, let :latest move to a
# new build, then roll back. Recording the TAG would restart the new image and
# report success. Recording the digest goes back to what was actually running.
rm -f /etc/hvym-img-tools/previous-image
echo "sha256:aaaa_the_old_build" > /tmp/latest_resolves_to
out=$(bash /work/update_proxy.sh latest 2>&1)
check "warned about the moving tag" "moving tag" "$out"
old_digest=$(cat /tmp/running_digest)
check "running the old build"       "sha256:aaaa_the_old_build" "$old_digest"

echo "sha256:bbbb_the_new_build" > /tmp/latest_resolves_to   # :latest moves
out=$(bash /work/update_proxy.sh 0.2.0 2>&1)                 # upgrade away from it
out=$(bash /work/update_proxy.sh --rollback 2>&1)
check "rolled back by digest"       "resolved to sha256:aaaa_the_old_build" "$out"
check "restored the OLD build"      "sha256:aaaa_the_old_build" "$(cat /tmp/running_digest)"

echo
echo "=========== RESULT ==========="
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
