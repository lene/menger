#!/usr/bin/env bash
# Fetches the Sprint 38 scene's environment map, Poly Haven's Kiara 2 Sunrise HDRI (CC0, Greg
# Zaal, https://polyhaven.com/a/kiara_2_sunrise), into assets/hdri/ (gitignored: the 4K/8K files
# are 26 / 102 MB and the staged-hygiene hook rejects files over 5 MB) and verifies it against a
# pinned SHA-256 (Poly Haven publishes MD5 only). Provenance: PROVENANCE.yaml.
#
# Use the file with: --texture-dir assets/hdri --env-map kiara_2_sunrise_<res>.hdr
#
# Usage: scripts/fetch-kiara-hdri.sh [1k|4k|8k]      (default 4k)
# Env (for tests): KIARA_HDRI_BASE_URL, KIARA_HDRI_DIR, KIARA_HDRI_PINS ("<res> <sha256>" lines)
# Exit: 0 verified file at the target; non-zero on unknown resolution, failed download or hash
# mismatch, leaving nothing at the target path.
set -euo pipefail

DEFAULT_RESOLUTION=4k
DEFAULT_BASE_URL=https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The SHA-256s of the files as downloaded on 2026-11-07; keep in sync with PROVENANCE.yaml
# (scripts/test-fetch-kiara-hdri.sh checks that).
DEFAULT_PINS="\
1k e4cd96c6046b2c3b81bd74b0c8b0d680272420ebc00e88ab7ba1c9be5941069b
4k f06d8b697c97cbaf4614ff82ae65530ed85ec6ab515bdfa4edaa6901ec2c0079
8k c6a1686cbcf45af26982af550a1c9647c5b24643c593e90d9cf394c3f4437d63"

PINS="${KIARA_HDRI_PINS:-$DEFAULT_PINS}"
BASE_URL="${KIARA_HDRI_BASE_URL:-$DEFAULT_BASE_URL}"
TARGET_DIR="${KIARA_HDRI_DIR:-$SCRIPT_DIR/../assets/hdri}"
RESOLUTION="${1:-$DEFAULT_RESOLUTION}"

pinned_resolutions() { awk '{print $1}' <<< "$PINS" | paste -sd' ' -; }
pin_for() { awk -v res="$1" '$1 == res {print $2}' <<< "$PINS"; }
sha256_of() { sha256sum "$1" | awk '{print $1}'; }

EXPECTED="$(pin_for "$RESOLUTION")"
if [ -z "$EXPECTED" ]; then
    echo "unknown resolution '$RESOLUTION'; pinned resolutions: $(pinned_resolutions)" >&2
    exit 2
fi

NAME="kiara_2_sunrise_${RESOLUTION}.hdr"
TARGET="$TARGET_DIR/$NAME"
URL="$BASE_URL/$RESOLUTION/$NAME"

if [ -f "$TARGET" ]; then
    actual="$(sha256_of "$TARGET")"
    if [ "$actual" = "$EXPECTED" ]; then
        echo "$TARGET already present, SHA-256 verified"
        exit 0
    fi
    echo "$TARGET has the wrong hash (want $EXPECTED, got $actual); downloading again" >&2
fi

mkdir -p "$TARGET_DIR"
TMP="$(mktemp "$TARGET_DIR/.$NAME.XXXXXX")"
trap 'rm -f "$TMP"' EXIT

echo "downloading $URL"
curl -fL --progress-bar -o "$TMP" "$URL"

actual="$(sha256_of "$TMP")"
if [ "$actual" != "$EXPECTED" ]; then
    # Not exposed under the target name; a stale corrupt copy must not stay either.
    rm -f "$TARGET"
    echo "checksum mismatch for $NAME (want $EXPECTED, got $actual)" >&2
    exit 1
fi

mv -f "$TMP" "$TARGET"
echo "$TARGET: SHA-256 verified"
