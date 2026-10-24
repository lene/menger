#!/usr/bin/env bash
# Checks that both containers that run generated scenes hold (review 2026-10-14, menger#78).
#
#   test-containment.sh flags   Docker-free: a fake `docker` on PATH records the arguments each
#                               wrapper really passes to `docker run`, and they are checked.
#                               Replaces a grep over the wrapper's text, which matched a comment
#                               and kept the nightly job red for a month.
#   test-containment.sh live    Runs hostile/probe-scene.scala through run-sandboxed.sh and, when
#                               an X display is reachable, run-render-window.sh, and requires
#                               every probe to be blocked. Needs the images (build.sh); honours
#                               MENGER_SCENE_VALIDATOR_IMAGE and MENGER_RENDER_IMAGE.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAILURES=0
fail() { echo "FAIL - $*"; FAILURES=$((FAILURES + 1)); }
ok() { echo "ok   - $*"; }

# require_args <label> <args-file> <flag>...: each flag must appear as an argument (a flag and
# its value may be one "--x y" string, matched as consecutive arguments).
require_args() {
  local label="$1" file="$2"; shift 2
  local joined; joined=" $(tr '\n' ' ' < "$file")"
  for flag in "$@"; do
    if [[ "$joined" == *" $flag "* ]]; then ok "$label passes $flag"; else fail "$label lacks $flag"; fi
  done
}
forbid_args() {
  local label="$1" file="$2"; shift 2
  for flag in "$@"; do
    if grep -qxF -- "$flag" "$file"; then fail "$label passes $flag"; else ok "$label omits $flag"; fi
  done
}

flags() {
  mkdir -p "$TMP/bin" "$TMP/session/window"
  cat > "$TMP/bin/docker" <<'EOF'
#!/bin/sh
case "$1" in
  run) shift; printf '%s\n' "$@" > "$DOCKER_ARGS_FILE" ;;
esac
exit 0
EOF
  chmod +x "$TMP/bin/docker"
  echo 'object S' > "$TMP/session/window/current.scala"
  local common=("--network none" "--cap-drop ALL" --read-only "--security-opt no-new-privileges"
    "--pull never" --pids-limit --memory --cpus)

  DOCKER_ARGS_FILE="$TMP/validator.args" PATH="$TMP/bin:$PATH" \
    "$HERE/run-sandboxed.sh" "$TMP/session/window/current.scala" >/dev/null 2>&1
  require_args "run-sandboxed.sh" "$TMP/validator.args" "${common[@]}"
  forbid_args "run-sandboxed.sh" "$TMP/validator.args" --gpus --device --privileged
  if grep -q '^type=bind,source=.*/window,' "$TMP/validator.args"; then
    fail "run-sandboxed.sh mounts a directory"
  else ok "run-sandboxed.sh mounts the scene file only"; fi

  # The window needs an X socket and a cookie; fake both so the wrapper reaches `docker run`.
  mkdir -p "$TMP/x"
  python3 -c "import socket,sys; s=socket.socket(socket.AF_UNIX); s.bind(sys.argv[1])" "$TMP/x/X42"
  local socket_dir=/tmp/.X11-unix
  if [ -S "$socket_dir/X42" ] || ln -s "$TMP/x/X42" "$socket_dir/X42" 2>/dev/null; then
    printf '#!/bin/sh\nexit 0\n' > "$TMP/bin/xauth"; chmod +x "$TMP/bin/xauth"
    DOCKER_ARGS_FILE="$TMP/window.args" PATH="$TMP/bin:$PATH" DISPLAY=:42 \
      "$HERE/run-render-window.sh" --scene "$TMP/session/window/current.scala" >/dev/null 2>&1
    rm -f "$socket_dir/X42"
    require_args "run-render-window.sh" "$TMP/window.args" "${common[@]}" \
      "--device nvidia.com/gpu=all"
    forbid_args "run-render-window.sh" "$TMP/window.args" --privileged --ipc=host --net=host
    if grep -qx "type=bind,source=$TMP/session/window,target=$TMP/session/window" "$TMP/window.args"
    then ok "run-render-window.sh mounts the window directory only"
    else fail "run-render-window.sh does not mount exactly the window directory"; fi
    if grep -q "source=$TMP/session," "$TMP/window.args"
    then fail "run-render-window.sh mounts the session directory"; fi
    if grep -q -- '-e' "$TMP/window.args" && grep -qiE '_KEY=|TOKEN=|SECRET=' "$TMP/window.args"
    then fail "run-render-window.sh forwards a credential"; else ok "run-render-window.sh forwards no credential"; fi
  else
    fail "cannot create a fake X socket in $socket_dir"
  fi
}

# probe_run <label> <stderr-file>: every PROBE line blocked, uid not 0, and all probes present.
check_probes() {
  local label="$1" log="$2"
  local expected=(network dns read-home read-sibling write-opt write-etc env-key docker-socket)
  for name in "${expected[@]}"; do
    line="$(grep -m1 "PROBE $name:" "$log")"
    case "$line" in
      *": blocked"*) ok "$label: $name blocked" ;;
      "") fail "$label: probe $name never ran" ;;
      *) fail "$label: $line" ;;
    esac
  done
  uid="$(grep -m1 'PROBE uid:' "$log" | sed 's/.*SUCCEEDED //')"
  if [ -n "$uid" ] && [ "$uid" != 0 ]; then ok "$label: runs as uid $uid"; else fail "$label: uid '$uid'"; fi
}

live() {
  mkdir -p "$TMP/session/window"
  echo "session secret" > "$TMP/session/history.jsonl"
  sed -e "s#@SIBLING@#$TMP/session/history.jsonl#" -e "s#@HOME@#$HOME#" \
    "$HERE/hostile/probe-scene.scala" > "$TMP/session/window/current.scala"

  ANTHROPIC_API_KEY=sk-containment-test \
    "$HERE/run-sandboxed.sh" "$TMP/session/window/current.scala" \
    --image "${MENGER_SCENE_VALIDATOR_IMAGE:-menger-scene-validator:latest}" \
    > "$TMP/validator.out" 2> "$TMP/validator.err"
  check_probes "validator" "$TMP/validator.err"

  local display_number="${DISPLAY#*:}"; display_number="${display_number%%.*}"
  if [ -n "${DISPLAY:-}" ] && [ -S "/tmp/.X11-unix/X$display_number" ]; then
    ANTHROPIC_API_KEY=sk-containment-test timeout --signal=TERM 60 \
      "$HERE/run-render-window.sh" --scene "$TMP/session/window/current.scala" \
      > "$TMP/window.out" 2> "$TMP/window.err"
    check_probes "render window" "$TMP/window.err"
  else
    echo "NOT CHECKED - render window: no X display reachable (DISPLAY='${DISPLAY:-}')"
  fi
}

case "${1:-}" in
  flags) flags ;;
  live) live ;;
  *) echo "usage: $0 flags|live" >&2; exit 2 ;;
esac
echo
if [ "$FAILURES" -eq 0 ]; then echo "all passed"; else echo "$FAILURES failed"; exit 1; fi
