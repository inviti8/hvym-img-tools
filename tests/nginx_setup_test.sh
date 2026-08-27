# setup_nginx.sh -- needs a real nginx, so run it in the nginx image, not ours:
#   docker run --rm -v "$PWD/tests:/tmp/h:ro" -v "$PWD/scripts:/work:ro" \
#     --entrypoint bash nginx:1.27 -c \
#     'apt-get -qq update >/dev/null && apt-get -qq install -y curl >/dev/null
#      bash /tmp/h/nginx_setup_test.sh'
#
# The harness stands up an `authen.test` neighbour first and checks it is still
# serving afterwards: this script runs on a VPS hosting a live project, so
# "did not break the neighbour" is the assertion that matters most.
set -u
PASS=0; FAIL=0
check() { # description, expected-substring, actual
  if printf '%s' "$3" | grep -qF "$2"; then echo "  [PASS] $1"; PASS=$((PASS+1));
  else echo "  [FAIL] $1"; echo "         wanted: $2"; echo "         got:    $(printf '%s' "$3" | tr '\n' '|' | head -c 400)"; FAIL=$((FAIL+1)); fi
}

# --- stand up a neighbour that must survive everything -----------------------
mkdir -p /var/www/authen /etc/nginx/sites-available /etc/nginx/sites-enabled
echo "AUTHEN IS ALIVE" > /var/www/authen/index.html
cat > /etc/nginx/sites-available/authen.test <<'A'
server {
    listen 80;
    server_name authen.test;
    root /var/www/authen;
    index index.html;
}
A
mkdir -p /etc/nginx/sites-enabled
ln -sf /etc/nginx/sites-available/authen.test /etc/nginx/sites-enabled/authen.test
grep -q "include /etc/nginx/sites-enabled" /etc/nginx/nginx.conf || \
  sed -i 's|include /etc/nginx/conf.d/\*.conf;|include /etc/nginx/conf.d/*.conf;\n    include /etc/nginx/sites-enabled/*;|' /etc/nginx/nginx.conf
rm -f /etc/nginx/conf.d/default.conf
printf '127.0.0.1 authen.test\n127.0.0.1 img.test\n' >> /etc/hosts
nginx -t >/dev/null 2>&1 || { echo "harness nginx config broken"; nginx -t; exit 1; }
nginx
sleep 1
echo "neighbour before: $(curl -s -m 5 http://authen.test/)"

echo
echo "=============== TEST 1: happy path ==============="
out=$(bash /work/setup_nginx.sh img.test --no-tls --upstream-port 8080 2>&1)
echo "$out" | sed 's/^/    /'
check "created the vhost"            "vhost written"          "$out"
check "config test passed"           "nginx -t passes"        "$out"
check "reloaded"                     "nginx reloaded"         "$out"
check "checked the neighbour"        "authen.test still HTTP" "$out"
check "warned proxy is absent"      "nothing is answering"   "$out"
check "proved configs untouched"    "no existing config file was modified" "$out"
check "took a backup"               "/etc/nginx backup"           "$out"
check "neighbour probe is real"     "authen.test -> HTTP 200" "$out"
check "vhost file exists"            "img.test"               "$(ls /etc/nginx/sites-available/)"
check "neighbour still serving"      "AUTHEN IS ALIVE"        "$(curl -s -m 5 http://authen.test/)"

echo
echo "=============== TEST 2: refuses to clobber a foreign config ==============="
out=$(bash /work/setup_nginx.sh authen.test --no-tls 2>&1)
check "refused to overwrite authen"  "not created by this script" "$out"
check "authen config intact"         "AUTHEN IS ALIVE"            "$(curl -s -m 5 http://authen.test/)"

echo
echo "=============== TEST 3: re-run over our own vhost is allowed ==============="
out=$(bash /work/setup_nginx.sh img.test --no-tls 2>&1)
check "recognised its own file"      "re-running over our own vhost" "$out"

echo
echo "=============== TEST 4: bad generated config rolls back ==============="
# a non-numeric timeout makes nginx -t fail, exercising the rollback path
out=$(bash /work/setup_nginx.sh bad.test --no-tls --read-timeout "not-a-number" 2>&1)
check "detected the bad config"      "config test FAILED"     "$out"
check "said it rolled back"          "rolled the new vhost back out" "$out"
check "removed the bad vhost"        ""                       "$(ls /etc/nginx/sites-available/ | grep -c bad.test || true)"
[ ! -f /etc/nginx/sites-available/bad.test ] && { echo "  [PASS] bad vhost file deleted"; PASS=$((PASS+1)); } || { echo "  [FAIL] bad vhost still present"; FAIL=$((FAIL+1)); }
nginx -t >/dev/null 2>&1 && { echo "  [PASS] nginx config still valid after rollback"; PASS=$((PASS+1)); } || { echo "  [FAIL] nginx config left broken"; FAIL=$((FAIL+1)); }
check "neighbour survived rollback"  "AUTHEN IS ALIVE"        "$(curl -s -m 5 http://authen.test/)"

echo
echo "=============== TEST 5: --rollback removes our vhost ==============="
out=$(bash /work/setup_nginx.sh img.test --rollback 2>&1)
check "rolled back cleanly"          "removed the vhost"      "$out"
[ ! -f /etc/nginx/sites-available/img.test ] && { echo "  [PASS] vhost file gone"; PASS=$((PASS+1)); } || { echo "  [FAIL] vhost file remains"; FAIL=$((FAIL+1)); }
check "neighbour still fine"         "AUTHEN IS ALIVE"        "$(curl -s -m 5 http://authen.test/)"

echo
echo "=============== TEST 6: refuses to run on an already-broken nginx ==============="
echo "this is not valid nginx syntax {{{" > /etc/nginx/sites-available/broken.test
ln -sf /etc/nginx/sites-available/broken.test /etc/nginx/sites-enabled/broken.test
out=$(bash /work/setup_nginx.sh img.test --no-tls 2>&1)
check "refused to proceed"           "ALREADY failing"        "$out"
rm -f /etc/nginx/sites-enabled/broken.test /etc/nginx/sites-available/broken.test

echo
echo "=============== TEST 7: --rollback refuses foreign files ==============="
out=$(bash /work/setup_nginx.sh authen.test --rollback 2>&1)
check "refused foreign rollback"     "refusing to remove"     "$out"
check "authen config still there"    "AUTHEN IS ALIVE"        "$(curl -s -m 5 http://authen.test/)"

[ "$FAIL" -eq 0 ] || exit 1

# --- --retune: change limits on a vhost certbot has since rewritten -----------
# The live box hit a 504 because 300s was below the mesh cold start. Rewriting
# the vhost to fix that would have discarded certbot's TLS block, so --retune
# edits the three directives in place. These assertions are about what it must
# NOT disturb.
echo
echo "=========== TEST: --retune ==========="

# An earlier case may have rolled our vhost back out, so start from a known one.
bash /work/setup_nginx.sh img.test --no-tls --force --upstream-port 18080 >/dev/null 2>&1

# Simulate certbot: add an ssl server block and a redirect to OUR vhost.
VH=/etc/nginx/sites-available/img.test
cat >> "$VH" <<'SSL'
server {
    listen 443 ssl; # managed by Certbot
    server_name img.test;
    ssl_certificate /etc/letsencrypt/live/img.test/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/img.test/privkey.pem; # managed by Certbot
    location / { proxy_pass http://127.0.0.1:18080; }
}
SSL
before_ssl=$(grep -c 'managed by Certbot' "$VH")
before_neighbour=$(curl -s -m 5 http://authen.test/ 2>/dev/null)

# 443 has no cert here, so drop that block for `nginx -t` while keeping the
# certbot markers we are asserting survive.
sed -i 's/^    listen 443 ssl; # managed by Certbot/    listen 8443; # managed by Certbot/' "$VH"
sed -i '/ssl_certificate/d' "$VH"
before_markers=$(grep -c 'managed by Certbot' "$VH")

out=$(bash /work/setup_nginx.sh img.test --retune --read-timeout 900 --max-upload 16 2>&1)
check "retuned the timeout"   "proxy_read_timeout 900s;" "$(cat $VH)"
check "retuned send timeout"  "proxy_send_timeout 900s;" "$(cat $VH)"
check "retuned the upload"    "client_max_body_size 16m;" "$(cat $VH)"
check "reported it"           "proxy_read/send_timeout 900s" "$out"
after_markers=$(grep -c 'managed by Certbot' "$VH")
[ "$after_markers" = "$before_markers" ] \
  && { echo "  [PASS] certbot's block survived ($after_markers markers)"; PASS=$((PASS+1)); } \
  || { echo "  [FAIL] certbot markers $before_markers -> $after_markers"; FAIL=$((FAIL+1)); }
check "neighbour still alive"  "AUTHEN IS ALIVE" "$(curl -s -m 5 http://authen.test/ 2>/dev/null)"
if printf '%s' "$out" | grep -q "==> TLS"; then
  echo "  [FAIL] --retune entered the TLS step; it must never touch certificates"; FAIL=$((FAIL+1))
else echo "  [PASS] never entered the TLS step"; PASS=$((PASS+1)); fi

echo
echo "=========== TEST: --retune refuses a foreign vhost ==========="
cat > /etc/nginx/sites-available/authen.test.bak2 <<'X'
server { listen 80; server_name nope.test; }
X
ln -sf /etc/nginx/sites-available/authen.test.bak2 /etc/nginx/sites-available/nope.test
out=$(bash /work/setup_nginx.sh nope.test --retune 2>&1)
check "refused"                "refusing to edit it" "$out"

echo
echo "=========== TEST: --retune needs an existing vhost ==========="
out=$(bash /work/setup_nginx.sh ghost.test --retune 2>&1)
check "told you to install"    "run without it first" "$out"

echo
echo "=========== FINAL RESULT ==========="
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
