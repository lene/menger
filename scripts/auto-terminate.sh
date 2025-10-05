#!/bin/bash
# Auto-terminate daemon for NVIDIA development spot instance
# Monitors SSH sessions and shuts down instance when all users log out

set -e

# Configuration
CHECK_INTERVAL=30          # Check every 30 seconds
GRACE_PERIOD=300          # Wait 5 minutes after last user logs out
LOG_FILE="/var/log/auto-terminate.log"

# Logging function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Get number of active SSH sessions
get_session_count() {
  # Count unique users with active SSH sessions (excluding the auto-terminate script itself)
  who | grep -c "pts/" || echo "0"
}

# Get instance metadata
get_instance_id() {
  curl -s http://169.254.169.254/latest/meta-data/instance-id
}

get_region() {
  curl -s http://169.254.169.254/latest/meta-data/placement/region
}

# Shutdown instance using AWS CLI
shutdown_instance() {
  local instance_id="$1"
  local region="$2"

  log "Initiating instance termination: $instance_id in $region"

  # Try to use AWS CLI to terminate
  if command -v aws &> /dev/null; then
    aws ec2 terminate-instances \
      --region "$region" \
      --instance-ids "$instance_id" 2>&1 | tee -a "$LOG_FILE"
  else
    log "AWS CLI not available, using shutdown command"
    sudo shutdown -h now
  fi
}

# Main daemon logic
main() {
  log "=== Auto-terminate daemon started ==="

  local instance_id
  local region

  # Get instance metadata
  instance_id=$(get_instance_id)
  region=$(get_region)

  log "Instance ID: $instance_id"
  log "Region: $region"
  log "Check interval: ${CHECK_INTERVAL}s"
  log "Grace period: ${GRACE_PERIOD}s"

  local last_session_count=0
  local no_session_since=0

  while true; do
    local session_count=$(get_session_count)

    if [ "$session_count" -gt 0 ]; then
      # Users are logged in
      if [ "$session_count" -ne "$last_session_count" ]; then
        log "Active sessions: $session_count"
      fi
      no_session_since=0
    else
      # No users logged in
      if [ "$no_session_since" -eq 0 ]; then
        no_session_since=$(date +%s)
        log "No active sessions detected. Grace period started."
      else
        local elapsed=$(($(date +%s) - no_session_since))
        if [ "$elapsed" -ge "$GRACE_PERIOD" ]; then
          log "Grace period expired (${elapsed}s). No sessions returned."
          shutdown_instance "$instance_id" "$region"
          exit 0
        elif [ $((elapsed % 60)) -eq 0 ]; then
          # Log every minute during grace period
          log "Grace period in progress: ${elapsed}s elapsed"
        fi
      fi
    fi

    last_session_count=$session_count
    sleep "$CHECK_INTERVAL"
  done
}

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
  echo "This script must be run as root or with sudo"
  exit 1
fi

# Create log file if it doesn't exist
touch "$LOG_FILE"
chmod 644 "$LOG_FILE"

# Run main daemon
main
