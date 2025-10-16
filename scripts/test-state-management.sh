#!/bin/bash
# Test state management scripts without requiring a running instance
# Uses mock data and local testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR=$(mktemp -d)
trap "rm -rf $TEST_DIR" EXIT

echo -e "${BLUE}=== State Management Test ===${NC}"
echo "Test directory: $TEST_DIR"
echo ""

PASSED=0
FAILED=0
WARNINGS=0

test_passed() {
  echo -e "${GREEN}✓${NC} $1"
  ((PASSED++))
}

test_failed() {
  echo -e "${RED}✗${NC} $1"
  ((FAILED++))
}

test_warning() {
  echo -e "${YELLOW}⚠${NC} $1"
  ((WARNINGS++))
}

# Test 1: Script Existence
echo -e "${BLUE}1. Script Files${NC}"
SCRIPTS=("backup-spot-state.sh" "restore-spot-state.sh" "list-spot-states.sh" "cleanup-spot-states.sh")
for script in "${SCRIPTS[@]}"; do
  if [ -f "$SCRIPT_DIR/$script" ]; then
    test_passed "$script exists"
  else
    test_failed "$script not found"
  fi
done
echo ""

# Test 2: Script Syntax
echo -e "${BLUE}2. Script Syntax${NC}"
for script in "${SCRIPTS[@]}"; do
  if [ -f "$SCRIPT_DIR/$script" ]; then
    if bash -n "$SCRIPT_DIR/$script" 2>/dev/null; then
      test_passed "$script syntax valid"
    else
      test_failed "$script has syntax errors"
    fi
  fi
done
echo ""

# Test 3: Script Executability
echo -e "${BLUE}3. Script Permissions${NC}"
for script in "${SCRIPTS[@]}"; do
  if [ -f "$SCRIPT_DIR/$script" ]; then
    if [ -x "$SCRIPT_DIR/$script" ]; then
      test_passed "$script is executable"
    else
      test_warning "$script not executable (should be chmod +x)"
    fi
  fi
done
echo ""

# Test 4: Required Tools
echo -e "${BLUE}4. Required Tools${NC}"
TOOLS=("rsync" "ssh" "scp" "jq" "tar" "du" "stat")
for tool in "${TOOLS[@]}"; do
  if command -v "$tool" &> /dev/null; then
    test_passed "$tool installed"
  else
    test_failed "$tool not found"
  fi
done
echo ""

# Test 5: Create Mock State Directory
echo -e "${BLUE}5. Mock State Creation${NC}"
export SPOT_STATES_DIR="$TEST_DIR/spot-states"
mkdir -p "$SPOT_STATES_DIR"

# Create mock state 1
STATE1="$SPOT_STATES_DIR/test-state-1"
mkdir -p "$STATE1/workspace/menger"
mkdir -p "$STATE1/config/fish"
mkdir -p "$STATE1/ssh"

echo "test file content" > "$STATE1/workspace/menger/test.txt"
echo "config content" > "$STATE1/config/fish/config.fish"
echo "ssh-rsa AAATEST" > "$STATE1/ssh/id_rsa.pub"

cat > "$STATE1/metadata.json" <<EOF
{
  "state_name": "test-state-1",
  "timestamp": "$(date -Iseconds -d '1 day ago' 2>/dev/null || date -Iseconds)",
  "instance_ip": "1.2.3.4",
  "instance_type": "g4dn.xlarge",
  "ami_id": "ami-test123",
  "git_branch": "main",
  "git_commit": "abc1234",
  "git_changed_files": 0,
  "backup_tool_version": "1.0"
}
EOF

SIZE1=$(du -sh "$STATE1" 2>/dev/null | cut -f1)
echo "$SIZE1" > "$STATE1/.size"

test_passed "Created mock state: test-state-1 ($SIZE1)"

# Create mock state 2 (older)
STATE2="$SPOT_STATES_DIR/test-state-2"
mkdir -p "$STATE2/workspace"
echo "old content" > "$STATE2/workspace/old.txt"

cat > "$STATE2/metadata.json" <<EOF
{
  "state_name": "test-state-2",
  "timestamp": "$(date -Iseconds -d '45 days ago' 2>/dev/null || date -Iseconds)",
  "instance_ip": "1.2.3.5",
  "git_branch": "feature-branch",
  "git_commit": "def5678",
  "git_changed_files": 3
}
EOF

SIZE2=$(du -sh "$STATE2" 2>/dev/null | cut -f1)
echo "$SIZE2" > "$STATE2/.size"

test_passed "Created mock state: test-state-2 ($SIZE2, 45 days old)"

# Create 'last' state
STATE_LAST="$SPOT_STATES_DIR/last"
mkdir -p "$STATE_LAST/workspace"
echo "last state" > "$STATE_LAST/workspace/file.txt"

cat > "$STATE_LAST/metadata.json" <<EOF
{
  "state_name": "last",
  "timestamp": "$(date -Iseconds)",
  "git_branch": "main",
  "git_commit": "xyz9999"
}
EOF

SIZE_LAST=$(du -sh "$STATE_LAST" 2>/dev/null | cut -f1)
echo "$SIZE_LAST" > "$STATE_LAST/.size"

test_passed "Created mock state: last ($SIZE_LAST)"
echo ""

# Test 6: List States Script
echo -e "${BLUE}6. List States Script${NC}"
if [ -x "$SCRIPT_DIR/list-spot-states.sh" ]; then
  OUTPUT=$("$SCRIPT_DIR/list-spot-states.sh" 2>/dev/null || echo "")

  if echo "$OUTPUT" | grep -q "test-state-1"; then
    test_passed "Lists test-state-1"
  else
    test_failed "Does not list test-state-1"
  fi

  if echo "$OUTPUT" | grep -q "test-state-2"; then
    test_passed "Lists test-state-2"
  else
    test_failed "Does not list test-state-2"
  fi

  if echo "$OUTPUT" | grep -q "last"; then
    test_passed "Lists 'last' state"
  else
    test_failed "Does not list 'last' state"
  fi

  if echo "$OUTPUT" | grep -q "auto-saved"; then
    test_passed "Highlights 'last' as auto-saved"
  else
    test_warning "'last' state not marked as auto-saved"
  fi

  if echo "$OUTPUT" | grep -q "Total states: 3"; then
    test_passed "Counts states correctly"
  else
    test_warning "State count mismatch"
  fi
else
  test_warning "list-spot-states.sh not executable, skipping"
fi
echo ""

# Test 7: Restore Script (List Mode)
echo -e "${BLUE}7. Restore Script (List Mode)${NC}"
if [ -x "$SCRIPT_DIR/restore-spot-state.sh" ]; then
  OUTPUT=$("$SCRIPT_DIR/restore-spot-state.sh" 2>/dev/null || echo "")

  if echo "$OUTPUT" | grep -q "Available Saved States"; then
    test_passed "Shows available states when no args"
  else
    test_failed "Does not show state list"
  fi

  if echo "$OUTPUT" | grep -q "test-state-1"; then
    test_passed "Lists states in restore mode"
  else
    test_failed "Does not list states"
  fi
else
  test_warning "restore-spot-state.sh not executable, skipping"
fi
echo ""

# Test 8: Cleanup Script (Dry-Run)
echo -e "${BLUE}8. Cleanup Script (Dry-Run)${NC}"
if [ -x "$SCRIPT_DIR/cleanup-spot-states.sh" ]; then
  # Test --older-than-days with dry-run
  OUTPUT=$("$SCRIPT_DIR/cleanup-spot-states.sh" --dry-run --older-than-days 30 2>/dev/null || echo "")

  if echo "$OUTPUT" | grep -q "test-state-2"; then
    test_passed "Identifies old state for deletion (>30 days)"
  else
    test_warning "Does not identify old state"
  fi

  if echo "$OUTPUT" | grep -q "DRY RUN"; then
    test_passed "Dry-run mode working"
  else
    test_failed "Dry-run mode not indicated"
  fi

  # Verify state still exists after dry-run
  if [ -d "$STATE2" ]; then
    test_passed "Dry-run does not delete files"
  else
    test_failed "Dry-run deleted files!"
  fi
else
  test_warning "cleanup-spot-states.sh not executable, skipping"
fi
echo ""

# Test 9: Cleanup Script (Keep-Recent)
echo -e "${BLUE}9. Cleanup Script (Keep-Recent)${NC}"
if [ -x "$SCRIPT_DIR/cleanup-spot-states.sh" ]; then
  OUTPUT=$("$SCRIPT_DIR/cleanup-spot-states.sh" --dry-run --keep-recent 2 2>/dev/null || echo "")

  if echo "$OUTPUT" | grep -q "States to delete: 1"; then
    test_passed "Identifies 1 state to remove (keeping 2)"
  else
    test_warning "Keep-recent filter may not be working correctly"
  fi
else
  test_warning "cleanup-spot-states.sh not executable, skipping"
fi
echo ""

# Test 10: Environment Variable Override
echo -e "${BLUE}10. Environment Variable Override${NC}"
CUSTOM_DIR="$TEST_DIR/custom-states"
mkdir -p "$CUSTOM_DIR"

export SPOT_STATES_DIR="$CUSTOM_DIR"
OUTPUT=$("$SCRIPT_DIR/list-spot-states.sh" 2>/dev/null || echo "")

if echo "$OUTPUT" | grep -q "No saved states found"; then
  test_passed "SPOT_STATES_DIR environment variable respected"
else
  test_warning "Environment variable override may not work"
fi

# Reset
export SPOT_STATES_DIR="$TEST_DIR/spot-states"
echo ""

# Test 11: Metadata Parsing
echo -e "${BLUE}11. Metadata Parsing${NC}"
if command -v jq &> /dev/null; then
  BRANCH=$(jq -r .git_branch "$STATE1/metadata.json" 2>/dev/null)
  if [ "$BRANCH" = "main" ]; then
    test_passed "Can parse git_branch from metadata"
  else
    test_failed "Metadata parsing failed"
  fi

  TIMESTAMP=$(jq -r .timestamp "$STATE1/metadata.json" 2>/dev/null)
  if [ -n "$TIMESTAMP" ] && [ "$TIMESTAMP" != "null" ]; then
    test_passed "Can parse timestamp from metadata"
  else
    test_failed "Timestamp parsing failed"
  fi
else
  test_warning "jq not available for metadata parsing tests"
fi
echo ""

# Test 12: Directory Structure Validation
echo -e "${BLUE}12. Directory Structure${NC}"
if [ -d "$STATE1/workspace" ]; then
  test_passed "workspace directory exists"
else
  test_failed "workspace directory missing"
fi

if [ -d "$STATE1/config" ]; then
  test_passed "config directory exists"
else
  test_failed "config directory missing"
fi

if [ -d "$STATE1/ssh" ]; then
  test_passed "ssh directory exists"
else
  test_failed "ssh directory missing"
fi

if [ -f "$STATE1/metadata.json" ]; then
  test_passed "metadata.json exists"
else
  test_failed "metadata.json missing"
fi
echo ""

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo -e "${GREEN}Passed:   $PASSED${NC}"
[ $WARNINGS -gt 0 ] && echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
[ $FAILED -gt 0 ] && echo -e "${RED}Failed:   $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ] && [ $WARNINGS -eq 0 ]; then
  echo -e "${GREEN}✓ All tests passed! State management system working.${NC}"
  exit 0
elif [ $FAILED -eq 0 ]; then
  echo -e "${YELLOW}✓ Tests passed with warnings. Review warnings above.${NC}"
  exit 0
else
  echo -e "${RED}✗ Some tests failed. Fix errors before using state management.${NC}"
  exit 1
fi
