#!/usr/bin/env bash
# The render window the scene agent shows accepted scenes in, run inside the same containment as
# validation (review 2026-10-14, menger#77). The window compiles and runs the generated scene, so
# running it on the host made "generated code only runs in the sandbox" false on the main path.
#
# A drop-in for menger-app's launcher: the same arguments, `--scene <file>` required. Adds what a
# window needs and nothing more: the GPU (CDI), one X display socket with a cookie for that
# display only, and the scene's own directory read-write at its host path (the agent replaces
# `current.scala` atomically, and the window writes `current.scala.window-state.json` next to it,
# so a single-file mount would go stale). Point the scene there from a directory holding nothing
# else -- the agent uses `<session>/window/`.
#
# Environment:
#   MENGER_RENDER_IMAGE                image (default menger-render-window:latest, see build.sh)
#   MENGER_RENDER_EXPECT_DSL_REVISION  refuse unless the image's DSL revision label matches
#   MENGER_TEXTURE_DIR                 mounted read-only at its host path, if set
#   MENGER_RENDER_MEMORY / _CPUS / _PIDS_LIMIT   resource bounds (12g / 4 / 1024)
set -euo pipefail

IMAGE="${MENGER_RENDER_IMAGE:-menger-render-window:latest}"
MEMORY="${MENGER_RENDER_MEMORY:-12g}"   # the launcher's -Xmx8g plus native and driver memory
CPUS="${MENGER_RENDER_CPUS:-4}"
PIDS="${MENGER_RENDER_PIDS_LIMIT:-1024}"

# shellcheck source=refused-result.sh
. "$(dirname "$(readlink -f "$0")")/refused-result.sh"   # refuse <message>: tagged JSON, exit 2

scene="" display="${DISPLAY:-}" lock=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    --scene) scene="${args[i + 1]:-}" ;;
    --display) display="${args[i + 1]:-}" ;;
    --render-lock-path) lock="${args[i + 1]:-}" ;;
  esac
done
[ -f "$scene" ] || refuse "render window: scene file not found: $scene"
scene_dir="$(cd "$(dirname "$scene")" && pwd)"

display_number="${display#*:}"; display_number="${display_number%%.*}"
[[ "$display_number" =~ ^[0-9]+$ ]] || refuse "render window: no X display (DISPLAY='$display')"
socket="/tmp/.X11-unix/X$display_number"
[ -S "$socket" ] || refuse "render window: X socket $socket not found"

if [ -n "${MENGER_RENDER_EXPECT_DSL_REVISION:-}" ]; then
  label="$(docker inspect -f '{{index .Config.Labels "menger.dsl-content-revision"}}' "$IMAGE" \
    2>/dev/null || true)"
  [ "$label" = "$MENGER_RENDER_EXPECT_DSL_REVISION" ] || refuse \
    "render image $IMAGE was built from a different DSL manifest (image: ${label:-<no label>}); rebuild it: menger/docker/scene-validator/build.sh"
fi

work="$(mktemp -d)"
name="menger-render-window-$$-$(date +%s)"
cleanup() {
  docker kill "$name" >/dev/null 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  rm -rf "$work"
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP

# A cookie for this display only, with the hostname wildcarded: the container has its own.
cookie="$work/xauth"
touch "$cookie"
xauth nlist "$display" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$cookie" nmerge - \
  || refuse "render window: no X authority cookie for display $display"
chmod 644 "$cookie"

mounts=(
  --mount "type=bind,source=$scene_dir,target=$scene_dir"
  --mount "type=bind,source=$socket,target=$socket,readonly"
  --mount "type=bind,source=$cookie,target=/xauth/cookie,readonly"
)
[ -z "${MENGER_TEXTURE_DIR:-}" ] \
  || mounts+=(--mount "type=bind,source=$MENGER_TEXTURE_DIR,target=$MENGER_TEXTURE_DIR,readonly")
[ -z "$lock" ] || mounts+=(--mount "type=bind,source=$(dirname "$lock"),target=$(dirname "$lock")")
case "$scene_dir${MENGER_TEXTURE_DIR:-}$lock" in
  *,*|*=*) refuse "render window: paths may not contain ',' or '='" ;;
esac

# Containment: as run-sandboxed.sh (no network, no capabilities, immutable root, bounded
# resources, never pull), plus the GPU and the display the window exists for. /tmp is exec:
# LWJGL and optix-jni extract their native libraries there.
docker run --rm --name "$name" \
  --pull never \
  --network none \
  --memory "$MEMORY" --cpus "$CPUS" --pids-limit "$PIDS" \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --read-only \
  --tmpfs /tmp:rw,exec,nosuid,size=2g \
  --device nvidia.com/gpu=all \
  --user "$(id -u):$(id -g)" \
  -e DISPLAY=":$display_number" -e XAUTHORITY=/xauth/cookie \
  -e __GL_THREADED_OPTIMIZATIONS=0 \
  `# OptiX compiles its shaders on the first frame, ~35 s; the immutable root and the tmpfs HOME` \
  `# lose its disk cache with the container. The scene directory is already mounted read-write` \
  `# and belongs to one agent session, so the cache lives there: later windows and exports of` \
  `# the session skip the compile, and no session reads another's (menger#133).` \
  -e MENGER_OPTIX_CACHE="$scene_dir/.optix-cache" \
  `# hybrid laptops: GLX on the NVIDIA GPU when the X server runs on the integrated one` \
  ${__NV_PRIME_RENDER_OFFLOAD:+-e __NV_PRIME_RENDER_OFFLOAD="$__NV_PRIME_RENDER_OFFLOAD"} \
  ${__GLX_VENDOR_LIBRARY_NAME:+-e __GLX_VENDOR_LIBRARY_NAME="$__GLX_VENDOR_LIBRARY_NAME"} \
  "${mounts[@]}" \
  "$IMAGE" "$@" &
wait $!
