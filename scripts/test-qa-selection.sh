#!/usr/bin/env bash
# Tests which suites qa-runner.sh selects for a given set of changed files, using the real
# qa-manifest.yaml and standards/rendering-paths.txt but stub suite scripts (no GPU, no sbt).
# Case: a diff that touches only build.sbt (an optix-jni pin bump) must select integration and
# smoke, because optix-jni renders every pixel (menger#144).
# Run: scripts/test-qa-selection.sh
set -u
# Run from the pre-push hook, git exports GIT_DIR (and friends) of the repository being pushed:
# left set, every `git -C "$repo"` below would act on that repository, not the scratch one
# (it committed the stub tree onto the branch being pushed and set a test identity in its
# config, 2026-11-15). Clear them all.
# shellcheck disable=SC2046
unset $(git rev-parse --local-env-vars)
# The gate's own exports, likewise: with QA_RANGES set, the runner below would take the pushed
# range (commits the scratch repo does not have) instead of CHANGED_FILES and select nothing.
unset QA_RANGES QA_REMOTE QA_INTERACTIVE

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
FAILURES=0

report() { # report <description> <exit-status>
    if [ "$2" -eq 0 ]; then echo "PASS: $1"; else echo "FAIL: $1"; FAILURES=$((FAILURES + 1)); fi
}

# stub_repo: a committed scratch repo with the real runner, manifest and path lists, where
# every suite of the manifest is a stub that reports PASS.
stub_repo() {
    local repo="$WORK/repo"
    mkdir -p "$repo/standards/scripts" "$repo/standards/hooks" "$repo/scripts/suites"
    cp "$ROOT/standards/scripts/qa-runner.sh" "$repo/standards/scripts/"
    cp "$ROOT/standards/hooks/lib.sh" "$repo/standards/hooks/"
    cp "$ROOT/standards/rendering-paths.txt" "$repo/standards/"
    cp "$ROOT/qa-manifest.yaml" "$repo/"
    local suite
    for suite in $(grep -oE '^  [a-z-]+:[[:space:]]*\{' "$ROOT/qa-manifest.yaml" | tr -d ' :{'); do
        printf '#!/bin/sh\necho "SUITE %s PASS n_failed=0 failed=\\"\\" reason=\\"\\""\n' "$suite" \
            > "$repo/scripts/suites/$suite.sh"
        chmod +x "$repo/scripts/suites/$suite.sh"
    done
    git -C "$repo" init --quiet
    git -C "$repo" config user.email test@example.com
    git -C "$repo" config user.name test
    git -C "$repo" add -A
    git -C "$repo" commit --quiet -m init
    echo "$repo"
}

# selected_suites <changed-files>: the suites qa-runner.sh invokes (not skipped) for that diff.
# The runner's resume cache reports a suite that passed on the same tree as "CACHED PASS"
# without invoking it, and every case runs on the same scratch tree: without clearing it, only
# the first case would see its suites run and every later case would select nothing.
selected_suites() {
    rm -rf "$REPO/.git/qa-cache"
    (cd "$REPO" && CHANGED_FILES="$1" ./standards/scripts/qa-runner.sh --tier push --all 2>&1) \
        | grep -E '^SUITE [a-z-]+ PASS' | cut -d' ' -f2 | sort | paste -sd' ' -
}

REPO="$(stub_repo)"
unset CI GITHUB_ACTIONS QA_ALLOW_ENV_SKIP

pin_bump="$(selected_suites 'build.sbt')"
echo "build.sbt only selects: $pin_bump"
for wanted in integration smoke; do
    case " $pin_bump " in
        *" $wanted "*) report "a diff of only build.sbt selects $wanted" 0 ;;
        *) report "a diff of only build.sbt selects $wanted" 1 ;;
    esac
done

readme_only="$(selected_suites 'README.md')"
case " $readme_only " in
    *" integration "* | *" smoke "*) report "a README-only diff selects neither integration nor smoke" 1 ;;
    *) report "a README-only diff selects neither integration nor smoke" 0 ;;
esac

# A diff of only a script that unit.sh tests must select unit (it was skipped on the push of
# scripts/check_shadow_tint.py and its test, menger#195, 2026-11-20).
for script in scripts/check_shadow_tint.py scripts/test_check_shadow_tint.py scripts/suites/unit.sh \
        scripts/render-cycle.sh scripts/caustics-validation/sponge_caustics.py; do
    case " $(selected_suites "$script") " in
        *" unit "*) report "a diff of only $script selects unit" 0 ;;
        *) report "a diff of only $script selects unit" 1 ;;
    esac
done

[ "$FAILURES" -eq 0 ] || { echo "$FAILURES failure(s)"; exit 1; }
echo "all passed"
