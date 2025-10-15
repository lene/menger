#!/bin/bash
# Cleanup old spot instance states
# Usage: cleanup-spot-states.sh [OPTIONS]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_BASE="${SPOT_STATES_DIR:-$HOME/.aws/spot-states}"
DRY_RUN=false
OLDER_THAN_DAYS=""
KEEP_RECENT=""
PROTECT_LAST=true
DELETE_STATE=""

# Usage function
usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Cleanup old spot instance states to free up disk space.

OPTIONS:
  --dry-run                 Show what would be deleted without deleting
  --older-than-days N       Delete states older than N days
  --keep-recent N           Keep only N most recent states
  --delete STATE_NAME       Delete specific state by name
  --no-protect-last         Allow deletion of 'last' auto-saved state
  -h, --help                Show this help message

EXAMPLES:
  # Show what would be deleted (dry run)
  $0 --dry-run --older-than-days 30

  # Delete states older than 30 days
  $0 --older-than-days 30

  # Keep only 5 most recent states
  $0 --keep-recent 5

  # Delete specific state
  $0 --delete old-experiment

  # Combination: keep 10 recent OR delete ones older than 60 days
  $0 --keep-recent 10 --older-than-days 60

EOF
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --older-than-days)
      OLDER_THAN_DAYS="$2"
      shift 2
      ;;
    --keep-recent)
      KEEP_RECENT="$2"
      shift 2
      ;;
    --delete)
      DELETE_STATE="$2"
      shift 2
      ;;
    --no-protect-last)
      PROTECT_LAST=false
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo -e "${RED}Error: Unknown option: $1${NC}"
      usage
      ;;
  esac
done

# Check if backup directory exists
if [ ! -d "$BACKUP_BASE" ]; then
  echo -e "${YELLOW}No states directory found${NC}"
  exit 0
fi

echo -e "${BLUE}=== Spot Instance State Cleanup ===${NC}"
[ "$DRY_RUN" = true ] && echo -e "${YELLOW}DRY RUN MODE - No files will be deleted${NC}"
echo ""

# Delete specific state
if [ -n "$DELETE_STATE" ]; then
  STATE_DIR="$BACKUP_BASE/$DELETE_STATE"

  if [ ! -d "$STATE_DIR" ]; then
    echo -e "${RED}Error: State not found: $DELETE_STATE${NC}"
    exit 1
  fi

  # Protect 'last' state
  if [ "$DELETE_STATE" = "last" ] && [ "$PROTECT_LAST" = true ]; then
    echo -e "${RED}Error: Cannot delete 'last' state (auto-saved)${NC}"
    echo "Use --no-protect-last flag to override"
    exit 1
  fi

  SIZE=$(du -sh "$STATE_DIR" 2>/dev/null | cut -f1)

  echo "State to delete: $DELETE_STATE"
  echo "Size:            $SIZE"
  echo ""

  if [ "$DRY_RUN" = false ]; then
    read -p "Are you sure you want to delete this state? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Cancelled"
      exit 0
    fi

    rm -rf "$STATE_DIR"
    echo -e "${GREEN}✓ State deleted: $DELETE_STATE${NC}"
  else
    echo -e "${YELLOW}Would delete: $DELETE_STATE ($SIZE)${NC}"
  fi

  exit 0
fi

# Collect state information
TMP_FILE=$(mktemp)
NOW_EPOCH=$(date +%s)

for state_dir in "$BACKUP_BASE"/*; do
  if [ -d "$state_dir" ]; then
    state=$(basename "$state_dir")

    # Get timestamp
    if [ -f "$state_dir/metadata.json" ]; then
      timestamp=$(jq -r .timestamp "$state_dir/metadata.json" 2>/dev/null || echo "unknown")
    else
      # Use directory modification time as fallback
      timestamp=$(stat -c %y "$state_dir" 2>/dev/null | cut -d' ' -f1 || echo "unknown")
    fi

    if [ "$timestamp" != "unknown" ]; then
      timestamp_epoch=$(date -d "$timestamp" +%s 2>/dev/null || echo "0")
    else
      timestamp_epoch=0
    fi

    # Get size
    size_bytes=$(du -sb "$state_dir" 2>/dev/null | cut -f1 || echo "0")
    size_human=$(du -sh "$state_dir" 2>/dev/null | cut -f1 || echo "unknown")

    # Calculate age in days
    age_seconds=$((NOW_EPOCH - timestamp_epoch))
    age_days=$((age_seconds / 86400))

    # Store: timestamp|state|age_days|size_bytes|size_human|timestamp_human
    echo "$timestamp_epoch|$state|$age_days|$size_bytes|$size_human|$timestamp" >> "$TMP_FILE"
  fi
done

# Check if any states found
STATE_COUNT=$(wc -l < "$TMP_FILE")
if [ $STATE_COUNT -eq 0 ]; then
  echo -e "${YELLOW}No states found${NC}"
  rm -f "$TMP_FILE"
  exit 0
fi

# Sort by timestamp (newest first)
sort -t'|' -k1 -rn "$TMP_FILE" -o "$TMP_FILE"

# Determine what to delete
TO_DELETE=$(mktemp)

# Apply --keep-recent filter
if [ -n "$KEEP_RECENT" ]; then
  echo -e "${BLUE}Applying keep-recent filter: keeping $KEEP_RECENT most recent states${NC}"

  tail -n +$((KEEP_RECENT + 1)) "$TMP_FILE" | while IFS='|' read -r timestamp_epoch state age_days size_bytes size_human timestamp_human; do
    # Protect 'last' state
    if [ "$state" = "last" ] && [ "$PROTECT_LAST" = true ]; then
      continue
    fi

    echo "$timestamp_epoch|$state|$age_days|$size_bytes|$size_human|$timestamp_human|keep-recent" >> "$TO_DELETE"
  done
fi

# Apply --older-than-days filter
if [ -n "$OLDER_THAN_DAYS" ]; then
  echo -e "${BLUE}Applying older-than filter: removing states older than $OLDER_THAN_DAYS days${NC}"

  while IFS='|' read -r timestamp_epoch state age_days size_bytes size_human timestamp_human; do
    if [ $age_days -gt $OLDER_THAN_DAYS ]; then
      # Protect 'last' state
      if [ "$state" = "last" ] && [ "$PROTECT_LAST" = true ]; then
        continue
      fi

      # Check if already marked for deletion
      if ! grep -q "^.*|$state|" "$TO_DELETE" 2>/dev/null; then
        echo "$timestamp_epoch|$state|$age_days|$size_bytes|$size_human|$timestamp_human|older-than" >> "$TO_DELETE"
      fi
    fi
  done < "$TMP_FILE"
fi

# Check if anything to delete
if [ ! -f "$TO_DELETE" ] || [ ! -s "$TO_DELETE" ]; then
  echo -e "${GREEN}No states to delete${NC}"
  echo ""
  echo "Current states:"
  scripts/list-spot-states.sh
  rm -f "$TMP_FILE" "$TO_DELETE"
  exit 0
fi

# Show what will be deleted
DELETE_COUNT=$(wc -l < "$TO_DELETE")
TOTAL_SIZE_BYTES=0

echo ""
echo -e "${YELLOW}States to delete: $DELETE_COUNT${NC}"
echo ""

sort -t'|' -k1 -rn "$TO_DELETE" | while IFS='|' read -r timestamp_epoch state age_days size_bytes size_human timestamp_human reason; do
  echo -e "  ${RED}✗${NC} $state"
  echo "     Saved: $age_days days ago ($timestamp_human)"
  echo "     Size:  $size_human"
  echo "     Reason: $reason"
  echo ""

  TOTAL_SIZE_BYTES=$((TOTAL_SIZE_BYTES + size_bytes))
done

# Calculate total size to free
TOTAL_SIZE_TO_FREE=0
while IFS='|' read -r timestamp_epoch state age_days size_bytes size_human timestamp_human reason; do
  TOTAL_SIZE_TO_FREE=$((TOTAL_SIZE_TO_FREE + size_bytes))
done < "$TO_DELETE"

if [ $TOTAL_SIZE_TO_FREE -gt 1073741824 ]; then
  # > 1 GB
  TOTAL_SIZE_HUMAN=$(echo "scale=2; $TOTAL_SIZE_TO_FREE / 1073741824" | bc)"G"
elif [ $TOTAL_SIZE_TO_FREE -gt 1048576 ]; then
  # > 1 MB
  TOTAL_SIZE_HUMAN=$(echo "scale=2; $TOTAL_SIZE_TO_FREE / 1048576" | bc)"M"
else
  TOTAL_SIZE_HUMAN="${TOTAL_SIZE_TO_FREE}B"
fi

echo -e "${BLUE}Total space to free: $TOTAL_SIZE_HUMAN${NC}"
echo ""

# Delete states
if [ "$DRY_RUN" = false ]; then
  read -p "Proceed with deletion? [y/N] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    rm -f "$TMP_FILE" "$TO_DELETE"
    exit 0
  fi

  echo ""
  echo -e "${YELLOW}Deleting states...${NC}"

  DELETED_COUNT=0
  while IFS='|' read -r timestamp_epoch state age_days size_bytes size_human timestamp_human reason; do
    STATE_DIR="$BACKUP_BASE/$state"
    if [ -d "$STATE_DIR" ]; then
      rm -rf "$STATE_DIR"
      echo "  ✓ Deleted: $state ($size_human)"
      DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
  done < "$TO_DELETE"

  echo ""
  echo -e "${GREEN}✓ Cleanup complete${NC}"
  echo "  Deleted: $DELETED_COUNT states"
  echo "  Freed:   $TOTAL_SIZE_HUMAN"
else
  echo -e "${YELLOW}DRY RUN - No files deleted${NC}"
fi

# Cleanup
rm -f "$TMP_FILE" "$TO_DELETE"

echo ""
echo "Remaining states:"
scripts/list-spot-states.sh 2>/dev/null || echo "(run scripts/list-spot-states.sh to see)"
