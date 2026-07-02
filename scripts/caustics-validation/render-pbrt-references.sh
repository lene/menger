#!/usr/bin/env bash
# Render committed pbrt-v4 caustics reference images (Sprint 33, Task 33.1).
#
# MANUAL / CI ONLY — never invoked by git hooks or the integration suite. Those
# read only the committed references in scripts/reference-images/caustics/.
#
# For each scene + resolution:
#   1. Render at the target sample budget (reference).
#   2. Render a "gold" at 2x budget; accept the reference only if
#      MSE(reference, gold) < CONVERGENCE_EPS (else warn — reference too noisy).
#   3. Convert EXR -> PFM (clipped linear, for metric comparison) and
#      EXR -> PNG (sRGB, for human viewing).
#   4. Write a manifest recording pbrt version, seed, spp, photons, scene SHA.
#
# Requires: pbrt and imgtool on PATH (installed at /usr/local/bin).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENES_DIR="$SCRIPT_DIR/scenes"
OUT_DIR="$SCRIPT_DIR/../reference-images/caustics"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PBRT="${PBRT:-/usr/local/bin/pbrt}"
IMGTOOL="${IMGTOOL:-/usr/local/bin/imgtool}"
SEED="${SEED:-0}"
SPP="${SPP:-256}"            # SPPM: pixelsamples == iteration count
NTHREADS="${NTHREADS:-0}"   # 0 => pbrt default (all cores)
CONVERGENCE_EPS="${CONVERGENCE_EPS:-0.0005}"

mkdir -p "$OUT_DIR"

# render_scene <scene-basename> <width> <height>
render_scene() {
  local scene="$1" width="$2" height="$3"
  local src="$SCENES_DIR/$scene.pbrt"
  local tag="${scene}_${width}x${height}"
  [ -f "$src" ] || { echo "MISSING scene: $src" >&2; return 1; }

  # Derive a resolution-specific scene into the throwaway work dir (source untouched).
  local scene_ref="$WORK_DIR/${tag}.pbrt"
  sed -e "s/\"integer xresolution\" [0-9]*/\"integer xresolution\" $width/" \
      -e "s/\"integer yresolution\" [0-9]*/\"integer yresolution\" $height/" \
      -e "s|\"string filename\" \"[^\"]*\"|\"string filename\" \"${tag}.exr\"|" \
      "$src" > "$scene_ref"

  local nthreads_arg=""
  [ "$NTHREADS" != "0" ] && nthreads_arg="--nthreads $NTHREADS"

  echo "== $tag : reference render (spp=$SPP) =="
  ( cd "$WORK_DIR" && "$PBRT" --seed "$SEED" --spp "$SPP" $nthreads_arg --quiet "$scene_ref" )

  echo "== $tag : gold render (spp=$((SPP * 2))) for convergence check =="
  local scene_gold="$WORK_DIR/${tag}_gold.pbrt"
  sed "s|\"string filename\" \"${tag}.exr\"|\"string filename\" \"${tag}_gold.exr\"|" \
      "$scene_ref" > "$scene_gold"
  ( cd "$WORK_DIR" && "$PBRT" --seed "$SEED" --spp "$((SPP * 2))" $nthreads_arg --quiet "$scene_gold" )

  # Convergence gate: MSE(reference, gold) must be small.
  # imgtool diff exits nonzero when images differ, so shield it from set -e/pipefail.
  local mse
  mse="$({ "$IMGTOOL" diff --metric MSE --reference "$WORK_DIR/${tag}_gold.exr" \
          "$WORK_DIR/${tag}.exr" 2>/dev/null || true; } | grep -oE 'MSE = [0-9.eE+-]+' | awk '{print $3}' || true)"
  echo "   convergence MSE(reference, gold) = ${mse:-?} (eps = $CONVERGENCE_EPS)"
  if [ -n "$mse" ] && awk "BEGIN{exit !($mse > $CONVERGENCE_EPS)}"; then
    echo "   WARNING: reference not converged at spp=$SPP; raise SPP." >&2
  fi

  # Convert for comparison (clipped linear PFM) and viewing (sRGB PNG).
  "$IMGTOOL" convert --clamp 1 --outfile "$OUT_DIR/${tag}.pfm" "$WORK_DIR/${tag}.exr"
  "$IMGTOOL" convert --outfile "$OUT_DIR/${tag}.png" "$WORK_DIR/${tag}.exr"

  # Manifest for reproducibility.
  {
    echo "scene:        $scene"
    echo "resolution:   ${width}x${height}"
    echo "pbrt_sha256:  $(sha256sum "$PBRT" | awk '{print $1}')"
    echo "seed:         $SEED"
    echo "spp:          $SPP"
    echo "convergence_mse: ${mse:-unknown}"
    echo "scene_sha256: $(sha256sum "$src" | awk '{print $1}')"
    echo "rendered:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$OUT_DIR/${tag}.manifest.txt"

  echo "   wrote $OUT_DIR/${tag}.{pfm,png,manifest.txt}"
}

# Resolutions to render, as "WxH" space-separated. Default: both the full-quality
# 800x600 reference and the 400x300 variant gated in the integration suite.
# Override e.g. RESOLUTIONS="400x300" for a fast gated-only regeneration.
RESOLUTIONS="${RESOLUTIONS:-800x600 400x300}"

SCENES=("${@:-canonical-caustics}")
for scene in "${SCENES[@]}"; do
  for res in $RESOLUTIONS; do
    render_scene "$scene" "${res%x*}" "${res#*x}"
  done
done

echo "Done. References in $OUT_DIR"
