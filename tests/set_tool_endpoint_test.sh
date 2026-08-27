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

cat > /usr/local/bin/docker <<'D'
#!/usr/bin/env bash
echo "docker $*" >> /tmp/docker.calls
case "$1" in
  restart) [ -f /tmp/restart_fails ] && exit 1; exit 0 ;;
esac
exit 0
D
chmod +x /usr/local/bin/docker

cat > /usr/local/bin/curl <<'C'
#!/usr/bin/env bash
[ -f /tmp/unhealthy ] && exit 7
# reflect whatever per-tool endpoints the env file currently declares
tools=$(grep -E '^RUNPOD_ENDPOINT_ID_' /etc/hvym-img-tools/proxy.env 2>/dev/null \
  | sed 's/^RUNPOD_ENDPOINT_ID_//; s/=.*//' | tr 'A-Z' 'a-z' | sed 's/.*/"&"/' | paste -sd, -)
printf '{"status":"ok","auth":true,"runpod_configured":true,"tool_endpoints":[%s]}' "$tools"
C
chmod +x /usr/local/bin/curl

echo "=========== TEST 1: set a tool endpoint ==========="
out=$(bash /work/set_tool_endpoint.sh mesh km99b7mrj2f85r 2>&1)
check "wrote the variable"     "RUNPOD_ENDPOINT_ID_MESH=km99b7mrj2f85r" "$(cat /etc/hvym-img-tools/proxy.env)"
check "restarted the proxy"    "docker restart" "$(cat /tmp/docker.calls)"
check "confirmed routing"      "routing 'mesh'" "$out"
check "backed up first"        "backed up to"   "$out"
check "SECRET SURVIVED"        "super-secret-must-survive" "$(cat /etc/hvym-img-tools/proxy.env)"
check "default untouched"      "RUNPOD_ENDPOINT_ID=reangle-ep" "$(cat /etc/hvym-img-tools/proxy.env)"

echo
echo "=========== TEST 2: --list ==========="
out=$(bash /work/set_tool_endpoint.sh --list 2>&1)
check "shows default"          "reangle-ep" "$out"
check "shows the tool"         "km99b7mrj2f85r" "$out"

echo
echo "=========== TEST 3: setting again replaces, does not duplicate ==========="
bash /work/set_tool_endpoint.sh mesh newid123 >/dev/null 2>&1
n=$(grep -c '^RUNPOD_ENDPOINT_ID_MESH=' /etc/hvym-img-tools/proxy.env)
[ "$n" = "1" ] && { echo "  [PASS] exactly one entry"; PASS=$((PASS+1)); } || { echo "  [FAIL] $n entries"; FAIL=$((FAIL+1)); }
check "value replaced"         "RUNPOD_ENDPOINT_ID_MESH=newid123" "$(cat /etc/hvym-img-tools/proxy.env)"

echo
echo "=========== TEST 4: a bad endpoint id is refused ==========="
out=$(bash /work/set_tool_endpoint.sh mesh 'bad id!' 2>&1)
check "rejected"               "looks wrong" "$out"
check "config unchanged"       "RUNPOD_ENDPOINT_ID_MESH=newid123" "$(cat /etc/hvym-img-tools/proxy.env)"

echo
echo "=========== TEST 5: unhealthy proxy rolls the config back ==========="
touch /tmp/unhealthy
out=$(bash /work/set_tool_endpoint.sh mesh wouldbreak 2>&1)
rm -f /tmp/unhealthy
check "detected"               "did not come back healthy" "$out"
check "rolled back"            "rolled back" "$out"
check "old value restored"     "RUNPOD_ENDPOINT_ID_MESH=newid123" "$(cat /etc/hvym-img-tools/proxy.env)"
check "SECRET STILL THERE"     "super-secret-must-survive" "$(cat /etc/hvym-img-tools/proxy.env)"

echo
echo "=========== TEST 6: failed restart restores config ==========="
touch /tmp/restart_fails
out=$(bash /work/set_tool_endpoint.sh mesh another 2>&1)
rm -f /tmp/restart_fails
check "detected"               "could not restart" "$out"
check "restored"               "RUNPOD_ENDPOINT_ID_MESH=newid123" "$(cat /etc/hvym-img-tools/proxy.env)"

echo
echo "=========== TEST 7: --remove ==========="
out=$(bash /work/set_tool_endpoint.sh --remove mesh 2>&1)
check "removed"                "Removing" "$out"
n=$(grep -c '^RUNPOD_ENDPOINT_ID_MESH=' /etc/hvym-img-tools/proxy.env || true)
[ "$n" = "0" ] && { echo "  [PASS] variable gone"; PASS=$((PASS+1)); } || { echo "  [FAIL] still present"; FAIL=$((FAIL+1)); }
check "secret survived removal" "super-secret-must-survive" "$(cat /etc/hvym-img-tools/proxy.env)"

echo
echo "=========== TEST 8: refuses without an installed proxy ==========="
mv /etc/hvym-img-tools/proxy.env /tmp/env.bak
out=$(bash /work/set_tool_endpoint.sh mesh x123 2>&1)
mv /tmp/env.bak /etc/hvym-img-tools/proxy.env
check "needs install first"    "run install_proxy.sh first" "$out"

echo
echo "=========== RESULT ==========="
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
