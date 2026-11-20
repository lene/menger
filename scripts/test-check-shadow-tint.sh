#!/usr/bin/env bash
# Tests check_shadow_tint.py offline on synthetic images (no GPU). Run: scripts/test-check-shadow-tint.sh
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
python3 -W error -m unittest -v test_check_shadow_tint
