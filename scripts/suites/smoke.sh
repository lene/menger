#!/bin/sh
# Suite: smoke (Sprint 36 B1). Sponge-type + DSL-animation smoke run, extracted verbatim
# from the CI run-smoke job — the same job never had a matching local hook step (a real
# B1/B4 gap; this closes both at once).
set -u
. ./standards/hooks/lib.sh

if [ "${HAS_RENDERING:-0}" -eq 0 ] && [ "${HAS_NATIVE:-0}" -eq 0 ]; then
  suite_skip smoke "no rendering or native changes"
  exit 0
fi

gpu_preflight_or_skip smoke || exit 1

STATUS=0
for spec in \
  "type=cube-sponge:level=1" "type=sponge-volume:level=1" \
  "type=sponge-volume:level=2" "type=tesseract" \
  "type=tesseract-sponge-volume:level=1" "type=tesseract-sponge-surface:level=2"; do
  if ! xvfb-run -a sbt "mengerApp / run --objects $spec --timeout 0.1"; then
    STATUS=1
    echo "smoke: FAILED - $spec"
  fi
done

if ! xvfb-run -a sbt "mengerApp / run --scene examples.dsl.PulsingSponge --frames 5 --start-t 0 --end-t 2 --headless --save-name menger%04d.png"; then
  STATUS=1
  echo "smoke: FAILED - PulsingSponge animation"
elif ! { test -f menger0000.png && test -f menger0004.png; }; then
  STATUS=1
  echo "smoke: FAILED - PulsingSponge animation did not produce expected frames"
fi

if [ "$STATUS" -eq 0 ]; then
  suite_pass smoke
else
  suite_fail smoke 1 "smoke-run"
fi
exit "$STATUS"
