#!/usr/bin/env bash
# Builds the scene-validator sandbox image (spec-ai-scene-agent story 5, AD-18).
#
# Two-step build, deliberately not a single `docker build .`:
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

echo "== 1/3: staging menger-app's distribution (sbt Universal/stage) =="
( cd "$REPO_ROOT" && sbt "mengerApp/Universal/stage" )

STAGE_SRC="$REPO_ROOT/menger-app/target/universal/stage"
if [ ! -d "$STAGE_SRC/lib" ]; then
  echo "error: $STAGE_SRC/lib not found -- did 'sbt Universal/stage' succeed?" >&2
  exit 1
fi

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

echo "== 3/3: docker build -t $IMAGE_TAG =="
docker build -t "$IMAGE_TAG" "$SCRIPT_DIR"

echo "Built $IMAGE_TAG"
