#!/usr/bin/env bash
# Tests the sponge-caustics comparison's helpers offline (the PLY generator and the fraction
# analysis; no GPU, no pbrt). Run: scripts/test-sponge-caustics.sh
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "${BASH_SOURCE[0]}")/caustics-validation" || exit 1
python3 -W error -m unittest -v test_sponge_caustics
