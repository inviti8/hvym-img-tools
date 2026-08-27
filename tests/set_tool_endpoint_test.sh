# set_tool_endpoint.sh -- run inside the proxy image:
#   docker run --rm -v "$PWD/tests:/tmp/h:ro" -v "$PWD/scripts:/work:ro" \
#     --entrypoint bash hvym-img-proxy:0.1.0 /tmp/h/set_tool_endpoint_test.sh
#
# The stubs model one thing carefully: docker snapshots --env-file at `run` and
# does NOT re-read it on `restart`. An earlier version of this harness had curl
# report whatever the env FILE said, so the suite passed green while the live
# proxy served `"tool_endpoints": null` -- the script was restarting a container
# that had never seen the edit. Here `docker run` copies the file into
# /tmp/container_env and `restart` leaves it alone, so only a real recreate
# makes healthz change.
set -u
PASS=0; FAIL=0
check() {
  if printf '%s' "$3" | grep -qF "$2"; then echo "  [PASS] $1"; PASS=$((PASS+1));
  else echo "  [FAIL] $1"; echo "         wanted: $2"; echo "         got: $(printf '%s' "$3" | tr '\n' '|' | head -c 300)"; FAIL=$((FAIL+1)); fi
}

mkdir -p /etc/hvym-img-tools
cat > /etc/hvym-img-tools/proxy.env <<'E'
RUNPOD_API_KEY=super-secret-must-survive
RUNPOD_ENDPOINT_ID=reangle-ep
HVYM_API_KEY=aaaaaaaaaaaaaaaaaaaaaaaa
HVYM_MAX_UPLOAD_MB=8
E
chmod 600 /etc/hvym-img-tools/proxy.env

# --- stub docker, modelling the behaviour that caused the bug ---------------
# `run` snapshots --env-file into the container config, exactly as docker does.
# `restart` does NOT re-read it. The script originally used restart, so every
# edit was silently ignored; the harness has to reproduce that to catch it.
cat > /usr/local/bin/docker <<'D'
#!/usr/bin/env bash
echo "docker $*" >> /tmp/docker.calls
case "$1" in
  run)
    [ -f /tmp/run_fails ] && exit 1
    grep -E '^RUNPOD_ENDPOINT_ID_' /etc/hvym-img-tools/proxy.env > /tmp/container_env 2>/dev/null || : > /tmp/container_env
    exit 0 ;;
  restart) [ -f /tmp/run_fails ] && exit 1; exit 0 ;;
  rm) exit 0 ;;
  inspect) echo "ghcr.io/inviti8/hvym-img-proxy:0.3.1"; exit 0 ;;
esac
exit 0
D
chmod +x /usr/local/bin/docker
: > /tmp/container_env

# --- stub curl: healthz reflects the CONTAINER's env, not the file ----------
cat > /usr/local/bin/curl <<'C'
#!/usr/bin/env bash
[ -f /tmp/unhealthy ] && exit 7
tools=$(sed 's/^RUNPOD_ENDPOINT_ID_//; s/=.*//' /tmp/container_env 2>/dev/null \
        | tr 'A-Z' 'a-z' | sed 's/.*/"&"/' | paste -sd, -)
printf '{"status":"ok","auth":true,"runpod_configured":true,"tool_endpoints":[%s]}' "$tools"
C
chmod +x /usr/local/bin/curl

echo "=========== TEST 1: set a tool endpoint ==========="
out=$(bash /work/set_tool_endpoint.sh mesh km99b7mrj2f85r 2>&1)
check "wrote the variable"      "RUNPOD_ENDPOINT_ID_MESH=km99b7mrj2f85r" "$(cat /etc/hvym-img-tools/proxy.env)"
check "RECREATED the container" "docker run" "$(cat /tmp/docker.calls)"
check "the container SAW it"    "km99b7mrj2f85r" "$(cat /tmp/container_env)"
check "confirmed routing"       "routing 'mesh'" "$out"
check "backed up first"         "backed up to"   "$out"
check "SECRET SURVIVED"         "super-secret-must-survive" "$(cat /etc/hvym-img-tools/proxy.env)"
check "default untouched"       "RUNPOD_ENDPOINT_ID=reangle-ep" "$(cat /etc/hvym-img-tools/proxy.env)"

echo
echo "=========== TEST 2: a plain restart would NOT have worked ==========="
# The regression itself: prove the script no longer relies on restart.
n=$(grep -c '^docker restart' /tmp/docker.calls || true)
[ "$n" = "0" ] && { echo "  [PASS] never used 'docker restart'"; PASS=$((PASS+1)); } \
                || { echo "  [FAIL] used 'docker restart' $n time(s) -- env edits would be ignored"; FAIL=$((FAIL+1)); }

echo
echo "=========== TEST 3: --list ==========="
out=$(bash /work/set_tool_endpoint.sh --list 2>&1)
check "shows default"           "reangle-ep" "$out"
check "shows the tool"          "km99b7mrj2f85r" "$out"

echo
echo "=========== TEST 4: setting again replaces, does not duplicate ==========="
bash /work/set_tool_endpoint.sh mesh newid123 >/dev/null 2>&1
n=$(grep -c '^RUNPOD_ENDPOINT_ID_MESH=' /etc/hvym-img-tools/proxy.env)
[ "$n" = "1" ] && { echo "  [PASS] exactly one entry"; PASS=$((PASS+1)); } || { echo "  [FAIL] $n entries"; FAIL=$((FAIL+1)); }
check "value replaced"          "RUNPOD_ENDPOINT_ID_MESH=newid123" "$(cat /etc/hvym-img-tools/proxy.env)"
check "container saw the new"   "newid123" "$(cat /tmp/container_env)"

echo
echo "=========== TEST 5: a bad endpoint id is refused ==========="
out=$(bash /work/set_tool_endpoint.sh mesh 'bad id!' 2>&1)
check "rejected"                "looks wrong" "$out"
check "config unchanged"        "RUNPOD_ENDPOINT_ID_MESH=newid123" "$(cat /etc/hvym-img-tools/proxy.env)"

echo
echo "=========== TEST 6: unhealthy proxy rolls the config back ==========="
touch /tmp/unhealthy
out=$(bash /work/set_tool_endpoint.sh mesh wouldbreak 2>&1)
rm -f /tmp/unhealthy
check "detected"                "did not come back healthy" "$out"
check "rolled back"             "rolled back" "$out"
check "old value restored"      "RUNPOD_ENDPOINT_ID_MESH=newid123" "$(cat /etc/hvym-img-tools/proxy.env)"
check "SECRET STILL THERE"      "super-secret-must-survive" "$(cat /etc/hvym-img-tools/proxy.env)"

echo
echo "=========== TEST 7: failed start restores config ==========="
touch /tmp/run_fails
out=$(bash /work/set_tool_endpoint.sh mesh another 2>&1)
rm -f /tmp/run_fails
check "detected"                "could not" "$out"
check "restored"                "RUNPOD_ENDPOINT_ID_MESH=newid123" "$(cat /etc/hvym-img-tools/proxy.env)"

echo
echo "=========== TEST 8: --remove ==========="
out=$(bash /work/set_tool_endpoint.sh --remove mesh 2>&1)
check "removed"                 "Removing" "$out"
n=$(grep -c '^RUNPOD_ENDPOINT_ID_MESH=' /etc/hvym-img-tools/proxy.env || true)
[ "$n" = "0" ] && { echo "  [PASS] variable gone"; PASS=$((PASS+1)); } || { echo "  [FAIL] still present"; FAIL=$((FAIL+1)); }
if [ ! -s /tmp/container_env ]; then echo "  [PASS] container saw the removal"; PASS=$((PASS+1));
else echo "  [FAIL] container still routes: $(cat /tmp/container_env)"; FAIL=$((FAIL+1)); fi
check "secret survived removal" "super-secret-must-survive" "$(cat /etc/hvym-img-tools/proxy.env)"

echo
echo "=========== TEST 9: refuses without an installed proxy ==========="
mv /etc/hvym-img-tools/proxy.env /tmp/env.bak
out=$(bash /work/set_tool_endpoint.sh mesh x123 2>&1)
mv /tmp/env.bak /etc/hvym-img-tools/proxy.env
check "needs install first"     "run install_proxy.sh first" "$out"

echo
echo "=========== RESULT ==========="
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
