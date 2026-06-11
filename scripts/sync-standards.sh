#!/bin/sh
# Sync canonical standards files into local checkouts of the sibling repos
# (Sprint 28.1). Copies only — never commits; review and push in each repo.
#
# usage: scripts/sync-standards.sh [path-to-menger-common] [path-to-optix-jni]
set -eu

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
STANDARDS_DIR="$REPO_ROOT/standards"
MANIFEST="$STANDARDS_DIR/manifest.txt"

# Resolve the main repository root even when running from a linked worktree,
# so sibling-repo defaults land next to the main checkout.
GIT_COMMON_DIR=$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir)
MAIN_REPO_ROOT=$(dirname "$GIT_COMMON_DIR")
WORKSPACE=$(dirname "$MAIN_REPO_ROOT")
TARGETS="${1:-$WORKSPACE/menger-common} ${2:-$WORKSPACE/optix-jni}"

manifest_entries() {
    grep -v '^[[:space:]]*\(#\|$\)' "$MANIFEST"
}

for target in $TARGETS; do
    if [ ! -d "$target/.git" ]; then
        echo "skip: $target is not a git checkout" >&2
        continue
    fi
    echo "=== syncing standards into $target ==="
    manifest_entries | while read -r repo_path canon_name; do
        if cmp -s "$STANDARDS_DIR/$canon_name" "$target/$repo_path"; then
            echo "unchanged: $repo_path"
        else
            cp "$STANDARDS_DIR/$canon_name" "$target/$repo_path"
            echo "updated:   $repo_path"
        fi
    done
    git -C "$target" status --short
done

echo
echo "Review and commit the changes in each sibling repo; nothing was committed."
