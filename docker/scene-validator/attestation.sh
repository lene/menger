# shellcheck shell=bash
# Sourced by run-sandboxed.sh and run-render-window.sh: the containment flags of `docker run` and
# the launcher's self-attestation (menger-scene-agent#58). The scene agent used to trust a launcher
# by its file name; now it runs `<launcher> --attest` and asserts what that prints.
#
# The attestation is built from the very argument list the launcher passes to `docker run`
# (CONTAINMENT, below), so it cannot say "network none" while the run says otherwise. What it does
# not prove: that the launcher is menger's (a script that prints a false attestation passes), and
# that the image it names is what a later run starts. The launcher paths come from the user's own
# environment: the attestation guards against a wrong or stale configuration (a host script, an
# old launcher without these flags), not against someone who controls the user's shell.
# contracts/process-result.json (`launcherAttestation`) is the specification of the line.

# shellcheck source=refused-result.sh
. "$(dirname "${BASH_SOURCE[0]}")/refused-result.sh"   # refuse, json_string

# containment_flags <memory> <cpus> <pids-limit> <tmpfs-options>: sets CONTAINMENT, the isolation
# flags of every `docker run` of a launcher. The one place they are spelled.
containment_flags() {
  CONTAINMENT=(
    --pull never                  # a missing image fails loudly, the daemon never fetches one
    --network none
    --memory "$1" --cpus "$2" --pids-limit "$3"
    --security-opt no-new-privileges
    --cap-drop ALL
    --read-only
    --tmpfs "/tmp:$4"
  )
}

# flag_value <flag> <args...>: the argument after <flag>, or nothing.
flag_value() {
  local flag="$1"; shift
  while [ $# -gt 1 ]; do
    [ "$1" != "$flag" ] || { printf '%s' "$2"; return 0; }
    shift
  done
}

# has_flag <flag> <args...>: whether <flag> is among the arguments.
has_flag() {
  local flag="$1"; shift
  local arg
  for arg in "$@"; do [ "$arg" != "$flag" ] || return 0; done
  return 1
}

# attestation <image> <docker run flags...>: the attestation line on stdout. Refuses (the tagged
# result, exit 2) when the image is not there: there is no digest to attest.
attestation() {
  local image="$1"; shift
  local network rootfs=writable digest
  network="$(flag_value --network "$@")"
  [ -n "$network" ] || network=default   # docker's own default network, i.e. not "none"
  ! has_flag --read-only "$@" || rootfs=read-only
  digest="$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null)" \
    || refuse "cannot attest: image $image not found (build it: menger/docker/scene-validator/build.sh)"
  printf '{"sandbox": "docker", "network": %s, "rootfs": %s, "image": %s, "image_digest": %s}\n' \
    "$(json_string "$network")" "$(json_string "$rootfs")" \
    "$(json_string "$image")" "$(json_string "$digest")"
}
