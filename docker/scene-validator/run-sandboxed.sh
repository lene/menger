#!/usr/bin/env bash
# AD-18's sandboxed run wrapper for the scene-validator image: GPU passthrough, no network
# egress, a restrictive seccomp profile, resource/wall-clock bounds, and a mount topology
# that matches AD-18 exactly -- scene workspace read-only, texture directory read-only,
# output read-write, and (this is load-bearing, not an omission) NO mount of any
# manifest/corpus artifact directory. AD-9 scopes the manifest+corpus artifact to the agent
# side only; a scene-validator run has no legitimate reason to read or write it, so nothing
# in this script ever bind-mounts it.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: run-sandboxed.sh <scene-file.scala> [--textures <dir>] [--output <dir>] [--image <tag>]

Runs `menger.tools.SceneValidator` against <scene-file.scala> inside the AD-18 sandbox.

  --textures <dir>   Mounted read-only at /workspace/textures (optional).
  --output <dir>     Mounted read-write at /workspace/output (optional; created if missing).
  --image <tag>      Image to run (default: menger-scene-validator:latest -- see build.sh).

Prints SceneValidator's tagged JSON result (ok | compile-errors | lint-findings | refused) to
stdout and exits non-zero for any tag other than `ok` (see SceneValidator.scala's `main`).
Also exits non-zero (with no JSON) if the run is killed by the wall-clock timeout below --
callers should treat "no valid JSON on stdout" as its own failure mode, not just a non-zero
exit.
USAGE
  exit 1
}

# Review round 1 findings, all fixed here: no resource/time bounds on the sandboxed process
# (a pathological or adversarial generated scene could hang or exhaust the host), and the
# scene's *entire parent directory* was mounted read-only rather than just the one file
# (any sibling file in that directory was readable from inside the sandbox, and could be
# copied into a writable --output mount -- an exfiltration path that doesn't need network
# egress at all, so --network none alone didn't close it).
TIMEOUT_SECONDS="${SCENE_VALIDATOR_TIMEOUT:-120}"
MEMORY_LIMIT="${SCENE_VALIDATOR_MEMORY:-2g}"
CPU_LIMIT="${SCENE_VALIDATOR_CPUS:-2}"
PIDS_LIMIT="${SCENE_VALIDATOR_PIDS_LIMIT:-256}"

[ $# -ge 1 ] || usage

SCENE_FILE="$1"; shift
TEXTURES_DIR=""
OUTPUT_DIR=""
IMAGE_TAG="menger-scene-validator:latest"

while [ $# -gt 0 ]; do
  case "$1" in
    --textures) TEXTURES_DIR="${2:?--textures needs a directory}"; shift 2 ;;
    --output)   OUTPUT_DIR="${2:?--output needs a directory}"; shift 2 ;;
    --image)    IMAGE_TAG="${2:?--image needs a tag}"; shift 2 ;;
    -h|--help)  usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[ -f "$SCENE_FILE" ] || { echo "Scene file not found: $SCENE_FILE" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_FILE_ABS="$(cd "$(dirname "$SCENE_FILE")" && pwd)/$(basename "$SCENE_FILE")"
SCENE_BASENAME="$(basename "$SCENE_FILE")"

# --- Mount topology (AD-18) --------------------------------------------------------------
# Binds the scene file itself, not its containing directory (review round 1 fix) -- a
# directory-level mount exposed every sibling file to the sandboxed process, an
# exfiltration path once paired with a writable --output mount.
MOUNT_ARGS=(--mount "type=bind,source=$SCENE_FILE_ABS,target=/workspace/scene/$SCENE_BASENAME,readonly")

if [ -n "$TEXTURES_DIR" ]; then
  [ -d "$TEXTURES_DIR" ] || { echo "Textures directory not found: $TEXTURES_DIR" >&2; exit 1; }
  TEXTURES_DIR_ABS="$(cd "$TEXTURES_DIR" && pwd)"
  MOUNT_ARGS+=(--mount "type=bind,source=$TEXTURES_DIR_ABS,target=/workspace/textures,readonly")
fi

if [ -n "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
  OUTPUT_DIR_ABS="$(cd "$OUTPUT_DIR" && pwd)"
  MOUNT_ARGS+=(--mount "type=bind,source=$OUTPUT_DIR_ABS,target=/workspace/output")
fi
# No manifest/corpus mount -- deliberately absent, see file header.

[ -f "$SCRIPT_DIR/seccomp-profile.json" ] || {
  echo "seccomp profile not found: $SCRIPT_DIR/seccomp-profile.json" >&2
  exit 1
}

# --- Containment (AD-18) -------------------------------------------------------------------
#   --pull never                        never implicitly fetch $IMAGE_TAG over the network if
#                                        it's not already cached locally -- `docker run` alone
#                                        would pull it via the *daemon's* own network access,
#                                        which happens before --network none applies to the
#                                        container being started, so an unbuilt/mistyped image
#                                        tag failed loudly instead of silently reaching out
#   --gpus all                          GPU passthrough via nvidia-container-toolkit
#   --network none                      no egress (AD-2: the renderer already needs none --
#                                        this is the enforcement, not a new constraint)
#   --memory / --cpus / --pids-limit    bounds a pathological or adversarial generated scene
#                                        (an infinite loop or huge allocation in a val
#                                        initializer, a fork bomb) to a fixed resource budget
#                                        instead of being able to exhaust the host
#   --security-opt seccomp=...          restricts syscalls (Docker's own default profile --
#                                        see seccomp-profile.json's own header comment for why)
#   --security-opt no-new-privileges    blocks setuid/setgid privilege escalation
#   --cap-drop ALL                      no Linux capabilities beyond the unprivileged default
#   --read-only + tmpfs /tmp            root filesystem is immutable; SceneCompiler's
#                                        Files.createTempDirectory("menger-scene-") writes to
#                                        java.io.tmpdir (/tmp), which is the one writable path
# `timeout` wraps the whole thing: a hung compile/construction (e.g. infinite recursion in a
# lazy val) is killed at the wall-clock bound even if it never hits the memory/CPU ceiling.
exec timeout --signal=KILL "${TIMEOUT_SECONDS}s" docker run --rm \
  --pull never \
  --gpus all \
  --network none \
  --memory "$MEMORY_LIMIT" \
  --cpus "$CPU_LIMIT" \
  --pids-limit "$PIDS_LIMIT" \
  --security-opt "seccomp=$SCRIPT_DIR/seccomp-profile.json" \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  "${MOUNT_ARGS[@]}" \
  "$IMAGE_TAG" \
  "/workspace/scene/$SCENE_BASENAME"
