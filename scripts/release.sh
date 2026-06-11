#!/bin/sh
# Release preparation for menger (Sprint 28.5).
# Updates all four version-carrying files and adds a CHANGELOG stub.
# Never creates commits or tags — that remains a conscious manual step.
#
# Usage: release.sh --prepare --version X.Y.Z
#        release.sh --check
#
# --prepare: bump version in the four files, insert CHANGELOG stub
# --check:   verify all four files carry the same version (exits non-zero on mismatch)
set -eu

command -v python3 > /dev/null || { echo "error: python3 required" >&2; exit 1; }

MODE=""
VERSION=""

while [ $# -gt 0 ]; do
  case "$1" in
    --prepare) MODE="prepare"; shift ;;
    --check)   MODE="check";   shift ;;
    --version) VERSION="$2";   shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[ -n "$MODE" ] || { echo "error: --prepare or --check required" >&2; exit 1; }

if [ "$MODE" = "prepare" ]; then
  [ -n "$VERSION" ] || {
    echo "error: --version X.Y.Z required for --prepare" >&2
    echo "       Version numbers are never inferred — you decide what the next version is." >&2
    exit 1
  }
  # Basic semver shape check
  echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$' || {
    echo "error: version must be X.Y.Z (e.g. 0.8.0), got: $VERSION" >&2
    exit 1
  }
fi

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# --- File locations (from check-version-consistency.sh) ------------------
BUILD_SBT="$REPO_ROOT/menger-app/build.sbt"
GITLAB_CI="$REPO_ROOT/.gitlab-ci.yml"
CLI_OPTIONS="$REPO_ROOT/menger-app/src/main/scala/menger/MengerCLIOptions.scala"
USER_GUIDE="$REPO_ROOT/docs/guide/user-guide.md"
USER_GUIDE_INDEX="$REPO_ROOT/docs/USER_GUIDE.md"
CHANGELOG="$REPO_ROOT/CHANGELOG.md"

for f in "$BUILD_SBT" "$GITLAB_CI" "$CLI_OPTIONS" "$USER_GUIDE" "$USER_GUIDE_INDEX" "$CHANGELOG"; do
  [ -f "$f" ] || { echo "error: required file not found: $f" >&2; exit 1; }
done

# --- check mode ----------------------------------------------------------
if [ "$MODE" = "check" ]; then
  exec "$REPO_ROOT/scripts/check-version-consistency.sh"
fi

# --- prepare mode --------------------------------------------------------
TODAY=$(python3 -c "import datetime; print(datetime.date.today())")

python3 - "$VERSION" "$TODAY" "$BUILD_SBT" "$GITLAB_CI" "$CLI_OPTIONS" \
           "$USER_GUIDE" "$USER_GUIDE_INDEX" "$CHANGELOG" << 'PYEOF'
import re
import sys

new_ver, today = sys.argv[1], sys.argv[2]
build_sbt, gitlab_ci, cli_opts, user_guide, user_guide_idx, changelog = sys.argv[3:]

def read(path):
    with open(path) as f:
        return f.read()

def write(path, content):
    with open(path, "w") as f:
        f.write(content)

# 1. menger-app/build.sbt
content = read(build_sbt)
updated = re.sub(r'(version\s*:=\s*")[^"]+(")', r'\g<1>' + new_ver + r'\2', content)
assert updated != content, "build.sbt: version pattern not found"
write(build_sbt, updated)
print(f"  build.sbt: version → {new_ver}")

# 2. .gitlab-ci.yml
content = read(gitlab_ci)
updated = re.sub(r'(DEPLOYABLE_VERSION:\s*)\S+', r'\g<1>' + new_ver, content)
assert updated != content, ".gitlab-ci.yml: DEPLOYABLE_VERSION pattern not found"
write(gitlab_ci, updated)
print(f"  .gitlab-ci.yml: DEPLOYABLE_VERSION → {new_ver}")

# 3. MengerCLIOptions.scala
content = read(cli_opts)
updated = re.sub(r'(version\("menger v)\d+\.\d+\.\d+', r'\g<1>' + new_ver, content)
assert updated != content, "MengerCLIOptions.scala: version string not found"
write(cli_opts, updated)
print(f"  MengerCLIOptions.scala: version string → {new_ver}")

# 4. docs/guide/user-guide.md
content = read(user_guide)
updated = re.sub(r'(?m)^(\*\*Version\*\*:\s*)\S+', r'\g<1>' + new_ver, content)
assert updated != content, "user-guide.md: **Version**: pattern not found"
write(user_guide, updated)
print(f"  docs/guide/user-guide.md: **Version** → {new_ver}")

# 5. docs/USER_GUIDE.md
content = read(user_guide_idx)
updated = re.sub(r'(?m)^(\*\*Version\*\*:\s*)\S+', r'\g<1>' + new_ver, content)
assert updated != content, "docs/USER_GUIDE.md: **Version**: pattern not found"
write(user_guide_idx, updated)
print(f"  docs/USER_GUIDE.md: **Version** → {new_ver}")

# 6. CHANGELOG.md — insert stub after the first heading line
content = read(changelog)
stub = f"""## [{new_ver}] - {today}

### Added
-

### Changed
-

### Fixed
-

"""
link_stub = f"[{new_ver}]: https://gitlab.com/lilacashes/menger/-/compare/PREV...{new_ver}\n"

# Insert stub after "# Changelog\n"
updated = re.sub(r'(# Changelog\n+)', r'\g<1>' + stub, content, count=1)
assert updated != content, "CHANGELOG.md: could not find insertion point"

# Append link at bottom (before the previous first link line)
if f"[{new_ver}]:" not in updated:
    updated = re.sub(r'(\[[\d.]+\]: https://)', link_stub + r'\1', updated, count=1)

write(changelog, updated)
print(f"  CHANGELOG.md: stub for [{new_ver}] inserted")
PYEOF

echo ""
echo "Version bumped to ${VERSION}. Next steps:"
echo "  1. Fill in the CHANGELOG.md entry (search for '## [${VERSION}]')"
echo "  2. Fill in the comparison URL in the CHANGELOG link (PREV → previous version tag)"
echo "  3. Review all five changed files:"
echo "       menger-app/build.sbt"
echo "       .gitlab-ci.yml"
echo "       menger-app/src/main/scala/menger/MengerCLIOptions.scala"
echo "       docs/guide/user-guide.md"
echo "       docs/USER_GUIDE.md"
echo "       CHANGELOG.md"
echo "  4. Commit and open an MR — merging without NORELEASE label triggers release."
echo "  5. Verify: scripts/release.sh --check"
