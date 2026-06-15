#!/bin/sh
# Standards parity / drift check (Sprint 28.1).
#
#   --local   verify this repo's copies match standards/ canon (fast, no network)
#   --remote  verify the GitHub sibling repos match standards/ canon (scheduled CI)
#
# Exit 0 = parity; exit 1 = drift (CI job fails loudly). With --remote and
# GITLAB_API_TOKEN set, a drift additionally opens a GitLab issue.
set -eu

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
STANDARDS_DIR="$REPO_ROOT/standards"
MANIFEST="$STANDARDS_DIR/manifest.txt"

GITHUB_OWNER="${GITHUB_OWNER:-lene}"
GITHUB_REPOS="${GITHUB_REPOS:-menger-common optix-jni}"
GITHUB_BRANCH="${GITHUB_BRANCH:-main}"
RAW_BASE="https://raw.githubusercontent.com"

usage() {
    echo "usage: $0 --local | --remote" >&2
    exit 2
}

[ $# -eq 1 ] || usage
MODE="$1"

drift=0

manifest_entries() {
    grep -v '^[[:space:]]*\(#\|$\)' "$MANIFEST"
}

check_local() {
    manifest_entries | while read -r repo_path canon_name; do
        if ! cmp -s "$REPO_ROOT/$repo_path" "$STANDARDS_DIR/$canon_name"; then
            echo "DRIFT (local): $repo_path differs from standards/$canon_name" >&2
            echo "fail" >> "$DRIFT_FLAG"
        else
            echo "ok (local): $repo_path"
        fi
    done
}

check_doc_versions() {
    # OPTIX_DOCKER_VERSION encodes "{CUDA}-{OptiX}-{Java}-{sbt}"; first field is CUDA.
    docker_ver=$(grep 'OPTIX_DOCKER_VERSION:' "$REPO_ROOT/.gitlab-ci.yml" \
        | sed 's/.*OPTIX_DOCKER_VERSION: *//' | sed 's/-.*//')
    apt_pkg="cuda-toolkit-$(echo "$docker_ver" | tr '.' '-')"
    if ! grep -q "$apt_pkg" "$REPO_ROOT/docs/INSTALLATION_FROM_SCRATCH.md"; then
        echo "DRIFT (local): CUDA version mismatch — CI uses CUDA $docker_ver but $apt_pkg not in docs/INSTALLATION_FROM_SCRATCH.md" >&2
        echo "fail" >> "$DRIFT_FLAG"
    else
        echo "ok (local): CUDA toolkit version consistent ($apt_pkg)"
    fi
}

check_remote() {
    for repo in $GITHUB_REPOS; do
        manifest_entries | while read -r repo_path canon_name; do
            url="$RAW_BASE/$GITHUB_OWNER/$repo/$GITHUB_BRANCH/$repo_path"
            tmp=$(mktemp)
            if ! curl --fail --silent --show-error --location -o "$tmp" "$url"; then
                echo "DRIFT (remote): $repo: cannot fetch $repo_path ($url)" >&2
                echo "fail" >> "$DRIFT_FLAG"
            elif ! cmp -s "$tmp" "$STANDARDS_DIR/$canon_name"; then
                echo "DRIFT (remote): $repo/$repo_path differs from standards/$canon_name" >&2
                echo "fail" >> "$DRIFT_FLAG"
            else
                echo "ok (remote): $repo/$repo_path"
            fi
            rm -f "$tmp"
        done
    done
}

open_gitlab_issue() {
    # Best-effort alarm escalation; the failing job is the primary signal.
    [ -n "${GITLAB_API_TOKEN:-}" ] || return 0
    [ -n "${CI_API_V4_URL:-}" ] || return 0
    [ -n "${CI_PROJECT_ID:-}" ] || return 0
    curl --silent --request POST \
        --header "PRIVATE-TOKEN: $GITLAB_API_TOKEN" \
        --data-urlencode "title=Standards drift detected ($(date -u +%Y-%m-%d))" \
        --data-urlencode "description=The scheduled StandardsDrift job found diverging standards files. See job $CI_JOB_URL and standards/README.md for the sync procedure." \
        --data-urlencode "labels=standards-drift" \
        "$CI_API_V4_URL/projects/$CI_PROJECT_ID/issues" > /dev/null || true
}

DRIFT_FLAG=$(mktemp)
trap 'rm -f "$DRIFT_FLAG"' EXIT

case "$MODE" in
    --local)  check_local; check_doc_versions ;;
    --remote) check_local; check_doc_versions; check_remote ;;
    *) usage ;;
esac

if [ -s "$DRIFT_FLAG" ]; then
    echo "Standards drift detected — see standards/README.md for the sync procedure." >&2
    [ "$MODE" = "--remote" ] && open_gitlab_issue
    exit 1
fi
echo "Standards parity verified ($MODE)."
