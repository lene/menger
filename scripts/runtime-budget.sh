#!/usr/bin/env bash
# Relative runtime budget: CPU user time of a level-1 render run of the staged app, divided by
# the user time of a `--help` run of the same launcher (JVM start, class loading, CLI parsing,
# no OptiX). Background load on the shared runner inflates both alike; an absolute 60 s budget
# failed at 69-108 s on unchanged code under desktop load (2026-10-03). The launcher, not
# `sbt run`: sbt load and the native build cost ~90 s of user time and drowned the app.
#
# Rounds alternate reference/subject order; the gate is the median of the per-round ratios.
# Needs `sbt mengerApp/stage` first.
# Usage: scripts/runtime-budget.sh   (env: RUNTIME_BUDGET_MAX_RATIO, RUNTIME_BUDGET_ROUNDS,
#        RUNTIME_BUDGET_REFERENCE_CMD, RUNTIME_BUDGET_SUBJECT_CMD to override for tests)
set -euo pipefail

MAX_RATIO="${RUNTIME_BUDGET_MAX_RATIO:?set RUNTIME_BUDGET_MAX_RATIO}"
ROUNDS="${RUNTIME_BUDGET_ROUNDS:-5}"
APP=./menger-app/target/universal/stage/bin/menger-app
REFERENCE_CMD="${RUNTIME_BUDGET_REFERENCE_CMD:-xvfb-run -a $APP --help}"
SUBJECT_CMD="${RUNTIME_BUDGET_SUBJECT_CMD:-xvfb-run -a $APP --objects type=sponge-volume:level=1 --timeout 0.1}"

TIMES="$(mktemp)"
trap 'rm -f "$TIMES"' EXIT

user_seconds() {
    /usr/bin/time -f %U -o "$TIMES" bash -c "$1" > /dev/null 2>&1
    cat "$TIMES"
}

ratios=()
for round in $(seq "$ROUNDS"); do
    if (( round % 2 )); then
        ref=$(user_seconds "$REFERENCE_CMD"); subj=$(user_seconds "$SUBJECT_CMD")
    else
        subj=$(user_seconds "$SUBJECT_CMD"); ref=$(user_seconds "$REFERENCE_CMD")
    fi
    ratio=$(echo "scale=3; $subj / $ref" | bc -l)
    echo "round $round: subject ${subj}s / reference ${ref}s = $ratio"
    ratios+=("$ratio")
done

median=$(printf '%s\n' "${ratios[@]}" | sort -g | awk '{a[NR]=$1} END {print (NR % 2) ? a[(NR+1)/2] : (a[NR/2] + a[NR/2+1]) / 2}')
echo "median ratio $median, limit $MAX_RATIO"
[ "$(echo "$median > $MAX_RATIO" | bc -l)" -eq 0 ]
