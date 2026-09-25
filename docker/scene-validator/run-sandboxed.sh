#!/usr/bin/env bash
# AD-18's sandboxed run wrapper for the scene-validator image: no network egress, no Linux
# capabilities, an immutable root filesystem, resource/wall-clock bounds, and a mount topology
# reduced to exactly one read-only file -- the scene under validation. Nothing else is mounted:
# not the texture directory, not an output directory, and (this is load-bearing, not an
# omission) no manifest/corpus artifact directory. AD-9 scopes the manifest+corpus artifact to
# the agent side only; a scene-validator run has no legitimate reason to read or write it.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: run-sandboxed.sh <scene-file.scala> [--image <tag>]

Runs `menger.tools.SceneValidator` against <scene-file.scala> inside the AD-18 sandbox.

  --image <tag>      Image to run (default: menger-scene-validator:latest -- see build.sh).

Prints SceneValidator's tagged JSON result (ok | compile-errors | lint-findings | refused) to
stdout. Exit codes mirror the validator's own: 0 for `ok`, 1 for a scene defect
(compile-errors/lint-findings), 2 for `refused`. A run killed by the wall-clock timeout below
exits 124 with no JSON -- callers should treat "no valid JSON on stdout" as its own failure
mode, not just a non-zero exit.
USAGE
  exit 1
}

# Review round 1 findings, both fixed here: no resource/time bounds on the sandboxed process
# (a pathological or adversarial generated scene could hang or exhaust the host), and the
# scene's *entire parent directory* was mounted read-only rather than just the one file
# (any sibling file in that directory was readable from inside the sandbox, and could be
# copied into a writable --output mount -- an exfiltration path that doesn't need network
# egress at all, so --network none alone didn't close it).
#
# Review round 2 findings, fixed here:
#   - `--textures` and `--output` are gone. Nothing consumed either: SceneValidator reads only
#     the scene file and writes only stdout. `--output` was a writable destination and
#     `--textures` a whole-directory mount, together reopening the very exfiltration shape the
#     single-file scene mount was introduced to close. The story that needs them re-adds them,
#     with the same read-only single-file discipline.
#   - `--gpus all` is gone. The image's own code path never touches the CUDA/OptiX bridge, so
#     handing untrusted generated code full device access bought nothing.
#   - The vendored seccomp profile is gone. It was Docker's stock default.json verbatim (428
#     syscalls, `socket`/`connect`/`execve`/`ptrace` all allowed), so `--security-opt seccomp=`
#     pointed at it was byte-identical in effect to passing nothing at all, while reading as a
#     control that existed. The real containment is enumerated below.
#   - `timeout` no longer wraps only the docker *client*. See the teardown block.
TIMEOUT_SECONDS="${SCENE_VALIDATOR_TIMEOUT:-120}"
MEMORY_LIMIT="${SCENE_VALIDATOR_MEMORY:-2g}"
CPU_LIMIT="${SCENE_VALIDATOR_CPUS:-2}"
PIDS_LIMIT="${SCENE_VALIDATOR_PIDS_LIMIT:-256}"

# Review round 2: `-h`/`--help` was handled in the option loop below, but never reached it --
# the first positional was consumed as the scene file unconditionally, so `--help` reported
# "Scene file not found: --help". Any leading flag is a usage error, not a filename.
[ $# -ge 1 ] || usage
case "$1" in
  -h|--help) usage ;;
  -*) echo "Expected a scene file, got option: $1" >&2; usage ;;
esac

SCENE_FILE="$1"; shift
IMAGE_TAG="menger-scene-validator:latest"

while [ $# -gt 0 ]; do
  case "$1" in
    --image)    IMAGE_TAG="${2:?--image needs a tag}"; shift 2 ;;
    -h|--help)  usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[ -f "$SCENE_FILE" ] || { echo "Scene file not found: $SCENE_FILE" >&2; exit 1; }

SCENE_FILE_ABS="$(cd "$(dirname "$SCENE_FILE")" && pwd)/$(basename "$SCENE_FILE")"
SCENE_BASENAME="$(basename "$SCENE_FILE")"

# Review round 2: `--mount` takes a comma-separated key=value list, so a path containing `,`
# or `=` is silently split into bogus mount options rather than mounting the file. Reject it
# with a real message instead.
case "$SCENE_FILE_ABS" in
  *,*|*=*) echo "Scene path may not contain ',' or '=': $SCENE_FILE_ABS" >&2; exit 1 ;;
esac

# --- Containment (AD-18) -------------------------------------------------------------------
#   --pull never                        never implicitly fetch $IMAGE_TAG over the network if
#                                        it's not already cached locally -- `docker run` alone
#                                        would pull it via the *daemon's* own network access,
#                                        which happens before --network none applies to the
#                                        container being started, so an unbuilt/mistyped image
#                                        tag fails loudly instead of silently reaching out
#   --network none                      no egress (AD-2: the renderer already needs none --
#                                        this is the enforcement, not a new constraint)
#   --memory / --cpus / --pids-limit    bounds a pathological or adversarial generated scene
#                                        (an infinite loop or huge allocation in a val
#                                        initializer, a fork bomb) to a fixed resource budget
#                                        instead of being able to exhaust the host
#   --security-opt no-new-privileges    blocks setuid/setgid privilege escalation
#   --cap-drop ALL                      no Linux capabilities beyond the unprivileged default
#   --read-only + tmpfs /tmp            root filesystem is immutable; SceneCompiler's
#                                        Files.createTempDirectory("menger-scene-") writes to
#                                        java.io.tmpdir (/tmp), which is the one writable path
#   single read-only bind mount         the scene file itself and nothing else
#   (the image additionally runs as a non-root user -- see Dockerfile's USER)
#
# Wall-clock teardown (review round 2): `timeout --signal=KILL ... docker run` sent SIGKILL to
# the docker *client*, which the containerd-supervised container outlives -- so a hung
# compile kept its memory, CPUs and PID budget indefinitely, exactly the outcome the bound
# exists to prevent, while the header claimed otherwise. The container now gets a name, the
# client is backgrounded, and the trap kills the container itself.
CONTAINER_NAME="menger-scene-validator-$$-$(date +%s)"

cleanup() {
  docker kill "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

set +e
timeout --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS}s" \
  docker run --rm \
  --name "$CONTAINER_NAME" \
  --pull never \
  --network none \
  --memory "$MEMORY_LIMIT" \
  --cpus "$CPU_LIMIT" \
  --pids-limit "$PIDS_LIMIT" \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --mount "type=bind,source=$SCENE_FILE_ABS,target=/workspace/scene/$SCENE_BASENAME,readonly" \
  "$IMAGE_TAG" \
  "/workspace/scene/$SCENE_BASENAME"
STATUS=$?
set -e

# `timeout` reports 124 on expiry. cleanup() (via the EXIT trap) stops the container itself,
# which is what the client being killed never did.
exit $STATUS
