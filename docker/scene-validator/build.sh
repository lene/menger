#!/usr/bin/env bash
# Builds the scene-validator sandbox image (spec-ai-scene-agent story 5, SA-AD-18).
#
# Three-step build, deliberately not a single `docker build .` (steps 1 and 2 below prepare
# the build context that step 3 consumes):
#   1. Stage menger-app's own distribution on the HOST (`sbt Universal/stage`) -- this is
#      where menger-geometry's native CUDA/CMake build already runs (see menger's own
#      build.sbt), producing libmengergeometry.so. Building that natively *inside* the image
#      would need the full CUDA devel toolkit + CMake + a licensed OptiX SDK checkout baked
#      into the image just to re-derive an artifact the host build already produced --
#      SceneValidator's own code path (compile -> load -> geometric-invariant check) never
#      calls into that native bridge at all, so there is nothing to gain from it.
#   2. Copy the host-built stage/ directory and a trimmed OptiX SDK (headers + the small
#      lib64/ directory only -- not the ~300MB of doc/ and SDK/ samples this image never
#      touches) into this directory as the Docker build context, then `docker build`.
#
# Usage: ./docker/scene-validator/build.sh [image-tag]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_TAG="${1:-menger-scene-validator:latest}"
OPTIX_ROOT="${OPTIX_ROOT:-/usr/local/optix}"

# Review round 2: docker was only reached at step 3, so a missing binary or a down daemon was
# discovered *after* the multi-minute sbt stage had already run. Fail in the first second.
command -v docker >/dev/null 2>&1 || {
  echo "error: docker not found on PATH" >&2
  exit 1
}
docker info >/dev/null 2>&1 || {
  echo "error: docker daemon unreachable -- is it running, and are you in the docker group?" >&2
  exit 1
}

# Stamped into the image as a label so a `menger-scene-validator:latest` that predates the
# working tree can be identified rather than silently validating scenes against an older DSL
# (review round 2). run-sandboxed.sh runs with `--pull never`, so a stale local image is the
# expected failure mode, not an exotic one.
APP_VERSION="$(grep -oP '(?<=version := ")[^"]+' "$REPO_ROOT/menger-app/build.sbt" | head -1)"
[ -n "$APP_VERSION" ] || {
  echo "error: could not read menger-app version from menger-app/build.sbt" >&2
  exit 1
}

echo "== 1/3: staging menger-app's distribution (sbt Universal/stage) =="
( cd "$REPO_ROOT" && sbt "mengerApp/Universal/stage" )

STAGE_SRC="$REPO_ROOT/menger-app/target/universal/stage"
if [ ! -d "$STAGE_SRC/lib" ]; then
  echo "error: $STAGE_SRC/lib not found -- did 'sbt Universal/stage' succeed?" >&2
  exit 1
fi

# F99: label the image with the manifest it was built with, so run-sandboxed.sh can refuse a
# caller whose manifest differs (the version label above cannot: it is the same all sprint).
MANIFEST_TMP="$(mktemp)"
"$STAGE_SRC/bin/manifest-generator" "$MANIFEST_TMP" >/dev/null
# The DSL's identity is the manifest's contentRevision, not the file's hash: a pin bump
# (optix-jni, Scala) changed the hash and made the agent refuse an image whose DSL had not
# changed (review 2026-10-14, menger#83).
DSL_REVISION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["contentRevision"])' "$MANIFEST_TMP")"
rm -f "$MANIFEST_TMP"
SOURCE_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"

echo "== 2/3: copying build context (stage/, optix-sdk/) =="
rm -rf "$SCRIPT_DIR/stage" "$SCRIPT_DIR/optix-sdk"
cp -a "$STAGE_SRC" "$SCRIPT_DIR/stage"

# Review round 1 fix: the two `if [ -d ... ]` guards below used to have no failure branch,
# so a misconfigured/unset OPTIX_ROOT silently produced empty optix-sdk/ directories and the
# build still reported success -- failing loudly here instead.
if [ ! -d "$OPTIX_ROOT/include" ]; then
  echo "error: OPTIX_ROOT/include not found at '$OPTIX_ROOT/include' -- set OPTIX_ROOT to a real OptiX SDK checkout" >&2
  exit 1
fi
if [ ! -d "$OPTIX_ROOT/lib64" ]; then
  echo "error: OPTIX_ROOT/lib64 not found at '$OPTIX_ROOT/lib64' -- set OPTIX_ROOT to a real OptiX SDK checkout" >&2
  exit 1
fi
mkdir -p "$SCRIPT_DIR/optix-sdk/include" "$SCRIPT_DIR/optix-sdk/lib64"
cp -a "$OPTIX_ROOT/include/." "$SCRIPT_DIR/optix-sdk/include/"
cp -a "$OPTIX_ROOT/lib64/." "$SCRIPT_DIR/optix-sdk/lib64/"

echo "== 3/3: docker build -t $IMAGE_TAG (menger-app $APP_VERSION) =="
# org.opencontainers.image.version lets a caller check the image against the working tree;
# menger.optix.redistributable=false records that this image embeds a licensed OptiX SDK
# (see the cp -a above) and must never be pushed to a public registry (review round 2).
build_target() {  # build_target <target> <tag>
  docker build --target "$1" \
    --label "org.opencontainers.image.version=$APP_VERSION" \
    --label "org.opencontainers.image.revision=$SOURCE_REVISION" \
    --label "menger.dsl-content-revision=$DSL_REVISION" \
    --label "org.opencontainers.image.title=menger-$1" \
    --label "menger.optix.redistributable=false" \
    -t "$2" "$SCRIPT_DIR"
}
build_target scene-validator "$IMAGE_TAG"
# The render window runs the same generated code (review 2026-10-14, menger#77); it is built
# from the same staged app in the same run, so validator and window are always one build and
# carry the same manifest label, which run-render-window.sh checks like run-sandboxed.sh.
RENDER_IMAGE_TAG="${RENDER_IMAGE_TAG:-menger-render-window:latest}"
build_target render-window "$RENDER_IMAGE_TAG"

echo "Built $IMAGE_TAG and $RENDER_IMAGE_TAG (menger-app $APP_VERSION, revision $SOURCE_REVISION, DSL revision $DSL_REVISION)"
echo "NOTE: this image embeds a licensed OptiX SDK -- do not push it to a public registry."
