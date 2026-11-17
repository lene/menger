#!/usr/bin/env bash
# Light a glass sponge gives back to the floor, menger vs pbrt-v4 (menger#150).
#
# Scene: level-2 sponge-volume, size 0.6, glass IOR 1.5, directional light 0.3,-1,0.6, floor at
# y=-0.5. Per renderer three renders (bare floor, opaque sponge, glass sponge) give
#   T = (glass - opaque) / (bare - opaque)
# per pixel; sponge_caustics.py prints the energy fraction returned inside the shadow footprint
# and over the whole floor. pbrt's twin uses BDPT (it can connect a light path that went through
# the glass to the camera); the sponge is a PLY from make_sponge_ply.py (menger's surface mesh).
#
# MANUAL / CI ONLY, never a git hook. Needs pbrt, a built menger-app, python3, and a GPU.
#
# Usage:  compare-sponge-caustics.sh [pbrt|menger|all|gate]      (default: all)
#   pbrt    only (re)render the pbrt side (CPU; needs no GPU)
#   menger  only (re)render the menger side, reusing an existing pbrt side
#   all     both, then the comparison table
#   gate    the caustics ladder's directional-light rung (integration-tests.sh): render only
#           menger's bare, opaque and caustics renders and check them against the committed pbrt
#           fractions in sponge-directional-reference.txt. Needs no pbrt.
#
# Environment (defaults in brackets):
#   MENGER_BIN        the menger-app launcher [menger-app on PATH]
#   PBRT              the pbrt-v4 binary [pbrt on PATH]
#   OUT_DIR           where renders and the PLY go; reused between runs [/tmp/sponge-caustics]
#   RUN_PREFIX        prefix for GPU commands, e.g. "flock /tmp/menger-gpu.lock" []
#   WIDTH HEIGHT      image size, both renderers [200 150]
#   PBRT_SPP          samples per pixel of the pbrt glass (BDPT) render [512; fractions are stable from 256 to 16384]
#   PBRT_THREADS      pbrt render threads [8]
#   PHOTONS ITERATIONS  menger photons per PPM iteration, iterations [500000 20]
#   MENGER_FLAGS      extra menger flags for the caustics render, e.g. "--accumulation-frames 4" []
#   MENGER_VARIANTS   menger glass variants to render [glass glass_ts glass_caustics]
#   MENGER_LIGHT_INTENSITY  menger light intensity [5]; keep the brightest pixel below 1 (clipped)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MENGER_BIN="${MENGER_BIN:-menger-app}"
PBRT="${PBRT:-pbrt}"
OUT_DIR="${OUT_DIR:-/tmp/sponge-caustics}"
RUN_PREFIX="${RUN_PREFIX:-}"
WIDTH="${WIDTH:-200}"
HEIGHT="${HEIGHT:-150}"
PBRT_SPP="${PBRT_SPP:-512}"
PBRT_THREADS="${PBRT_THREADS:-8}"
PHOTONS="${PHOTONS:-500000}"
ITERATIONS="${ITERATIONS:-20}"
MENGER_FLAGS="${MENGER_FLAGS:-}"
MENGER_VARIANTS="${MENGER_VARIANTS:-glass glass_ts glass_caustics}"

# The scene (the issue's, verbatim) and the view of it.
LEVEL=2
SIZE=0.6
IOR=1.5
LIGHT_DIR="0.3,-1,0.6"
FLOOR_Y=-0.5
FLOOR_EXTENT=2              # half-width of the pbrt floor quad; a small floor keeps BDPT's distant
                            # light from wasting samples far from the sponge
# A dark floor: pbrt's glass render also carries the light the floor sends up into the sponge and
# the glass sends back down (floor, glass, floor), which menger's caustics do not have (they are
# light, glass, floor). At albedo 0.5 that adds 9% (footprint) and 23% (whole floor) to pbrt's
# fractions; at 0.1 it is under 2%. FLOOR_COLOR is the same albedo as menger's linear grey.
FLOOR_COLOR="${FLOOR_COLOR:-#1A1A1A}"
FLOOR_ALBEDO="${FLOOR_ALBEDO:-0.102}"
MENGER_LIGHT_INTENSITY="${MENGER_LIGHT_INTENSITY:-5}"  # direct light well above menger's constant ambient fill, no pixel clipped
PBRT_LIGHT_L=3
EYE="1.0,2.2,1.6"           # looks at the shadow from the side it falls on, so the sponge
LOOKAT="0.1,-0.5,0.2"       # hides little of it
UP="0,1,0"

PBRT_BARE_OPAQUE_SPP=64     # no caustics to find: a path tracer is exact and cheap
PBRT_MAX_DEPTH=50           # a photon crosses many walls of a level-2 sponge

mkdir -p "$OUT_DIR"

vertical_fov() { # pbrt applies fov to the shorter axis; menger's 45 deg is horizontal
  python3 -c "import math; print(2*math.degrees(math.atan(math.tan(math.radians(22.5))*$HEIGHT/$WIDTH)))"
}

# write_pbrt_scene <variant> <spp> <integrator-block> <sponge-block>
write_pbrt_scene() {
  local variant="$1" spp="$2" integrator="$3" sponge="$4"
  python3 - "$SCRIPT_DIR/scenes/sponge-directional.pbrt.in" "$OUT_DIR/pbrt_$variant.pbrt" <<PY
import sys
src, dst = sys.argv[1:]
subs = {
  "@OUTFILE@": "pbrt_$variant.pfm", "@WIDTH@": "$WIDTH", "@HEIGHT@": "$HEIGHT",
  "@EYE@": "${EYE//,/ }", "@LOOKAT@": "${LOOKAT//,/ }", "@FOV@": "$(vertical_fov)", "@SPP@": "$spp",
  "@LIGHT_DIR@": "${LIGHT_DIR//,/ }", "@LIGHT_L@": "$PBRT_LIGHT_L", "@FLOOR_Y@": "$FLOOR_Y",
  "@FLOOR_EXTENT@": "$FLOOR_EXTENT", "@NEG_FLOOR_EXTENT@": "-$FLOOR_EXTENT",
  "@FLOOR_ALBEDO@": "$FLOOR_ALBEDO",
  "@INTEGRATOR@": """$integrator""", "@SPONGE@": """$sponge""",
}
text = open(src).read()
for key, value in subs.items():
  text = text.replace(key, value)
open(dst, "w").write(text)
PY
}

render_pbrt() {
  python3 "$SCRIPT_DIR/make_sponge_ply.py" "$LEVEL" "$SIZE" "$OUT_DIR/sponge.ply"
  local path="Integrator \"path\"
    \"integer maxdepth\" $PBRT_MAX_DEPTH"
  local bdpt="Integrator \"bdpt\"
    \"integer maxdepth\" $PBRT_MAX_DEPTH"
  local mesh='Shape "plymesh" "string filename" "sponge.ply"'
  write_pbrt_scene bare   "$PBRT_BARE_OPAQUE_SPP" "$path" ""
  write_pbrt_scene opaque "$PBRT_BARE_OPAQUE_SPP" "$path" \
    "AttributeBegin
    Material \"diffuse\" \"rgb reflectance\" [ 0 0 0 ]
    $mesh
AttributeEnd"
  write_pbrt_scene glass  "$PBRT_SPP" "$bdpt" \
    "AttributeBegin
    Material \"dielectric\" \"float eta\" $IOR
    $mesh
AttributeEnd"
  local variant
  for variant in bare opaque glass; do
    echo "== pbrt $variant =="
    ( cd "$OUT_DIR" && time "$PBRT" --nthreads "$PBRT_THREADS" --quiet "pbrt_$variant.pbrt" )
  done
}

# render_menger <variant> <objects-spec> [flags...]
render_menger() {
  local variant="$1" objects="$2"
  shift 2
  echo "== menger $variant =="
  # shellcheck disable=SC2086
  $RUN_PREFIX "$MENGER_BIN" --headless --allow-uniform-render --save-name "$OUT_DIR/menger_$variant.pfm" \
    --width "$WIDTH" --height "$HEIGHT" \
    --plane "y:$FLOOR_Y" --plane-color "$FLOOR_COLOR" \
    --light "directional:$LIGHT_DIR:$MENGER_LIGHT_INTENSITY" --shadows --ambient constant \
    --camera-pos "$EYE" --camera-lookat "$LOOKAT" --camera-up "$UP" \
    --objects "$objects" "$@" 2>&1 | { grep -iE "error|exception" || true; }
}

render_menger_all() {
  local glass="type=sponge-volume:level=$LEVEL:size=$SIZE:material=glass:ior=$IOR"
  render_menger bare   "type=sphere:size=0.001:pos=0,9,0"
  render_menger opaque "type=sponge-volume:level=$LEVEL:size=$SIZE:color=#000000"
  local variant
  for variant in $MENGER_VARIANTS; do
    case "$variant" in
      glass)          render_menger glass "$glass" ;;
      glass_ts)       render_menger glass_ts "$glass" --transparent-shadows ;;
      glass_caustics) # shellcheck disable=SC2086
                      render_menger glass_caustics "$glass" --caustics \
                        --caustics-photons "$PHOTONS" --caustics-iterations "$ITERATIONS" $MENGER_FLAGS ;;
      *) echo "unknown variant $variant" >&2; return 1 ;;
    esac
  done
}

report() {
  local triples=(--triple pbrt "$OUT_DIR/pbrt_bare.pfm" "$OUT_DIR/pbrt_opaque.pfm" "$OUT_DIR/pbrt_glass.pfm")
  local variant
  for variant in $MENGER_VARIANTS; do
    triples+=(--triple "menger $variant" "$OUT_DIR/menger_bare.pfm" "$OUT_DIR/menger_opaque.pfm" "$OUT_DIR/menger_$variant.pfm")
  done
  python3 "$SCRIPT_DIR/sponge_caustics.py" --eye "$EYE" --lookat "$LOOKAT" --up "$UP" \
    --size "$SIZE" --floor-y "$FLOOR_Y" --extent "$FLOOR_EXTENT" "${triples[@]}"
}

# gate: the pbrt fractions are committed (sponge-directional-reference.txt), not re-rendered
gate() {
  local reference="$SCRIPT_DIR/sponge-directional-reference.txt" fp all tolerance
  fp="$(awk '$1 == "fp" {print $2}' "$reference")"
  all="$(awk '$1 == "all" {print $2}' "$reference")"
  tolerance="$(awk '$1 == "tolerance" {print $2}' "$reference")"
  MENGER_VARIANTS="glass_caustics"
  render_menger_all
  python3 "$SCRIPT_DIR/sponge_caustics.py" --eye "$EYE" --lookat "$LOOKAT" --up "$UP" \
    --size "$SIZE" --floor-y "$FLOOR_Y" --extent "$FLOOR_EXTENT" \
    --triple "menger caustics" "$OUT_DIR/menger_bare.pfm" "$OUT_DIR/menger_opaque.pfm" \
             "$OUT_DIR/menger_glass_caustics.pfm" \
    --gate "menger caustics" "$fp" "$all" "$tolerance"
}

case "${1:-all}" in
  gate)   gate ;;
  pbrt)   render_pbrt ;;
  menger) render_menger_all; report ;;
  all)    render_pbrt; render_menger_all; report ;;
  *)      echo "usage: $0 [pbrt|menger|all|gate]" >&2; exit 2 ;;
esac
