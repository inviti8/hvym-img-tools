#!/usr/bin/env bash
# RunPod proxy SSH gives an interactive shell and IGNORES the remote command arg,
# so commands go over stdin. `stty -echo` stops the PTY echoing input back at us.
HOST="${POD_HOST:?}"; TMO="${TMO:-300}"
{ printf 'stty -echo 2>/dev/null\nset -o pipefail\n'; printf '%s\n' "$1"; printf 'exit\n'; } \
  | timeout "$TMO" ssh -tt \
  -i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=10 -o LogLevel=ERROR "$HOST" 2>&1 \
  | tr -d '\r' | sed -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' -e 's/\x1b\][0-9];[^\x07]*\x07//g'
