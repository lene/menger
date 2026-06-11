#!/bin/sh
# Install the CI-runner hardening (menger Sprint 28.3): resource slice and
# service drop-ins. Idempotent. Run as the desktop user; escalates via pkexec.
#
# Does NOT restart the services — do that manually when no CI jobs are running:
#   pkexec systemctl restart gitlab-runner.service
#   pkexec systemctl restart 'actions.runner.*.service'
#
# GitLab job-container caps (cpus/memory in /etc/gitlab-runner/config.toml)
# are applied separately — see README.md; that file holds runner tokens and is
# never managed from this repository.
set -eu

DIR=$(cd "$(dirname "$0")" && pwd)
GITHUB_UNIT=$(systemctl list-units --type=service --all --no-legend 'actions.runner.*' \
    | awk '{print $1}' | head -n 1)

echo "Installing ci-runners.slice and drop-ins (pkexec will prompt)..."
pkexec sh -c "
    set -eu
    install -m 644 '$DIR/ci-runners.slice' /etc/systemd/system/ci-runners.slice
    install -d /etc/systemd/system/gitlab-runner.service.d
    install -m 644 '$DIR/gitlab-runner-override.conf' /etc/systemd/system/gitlab-runner.service.d/override.conf
    if [ -n '$GITHUB_UNIT' ]; then
        install -d '/etc/systemd/system/$GITHUB_UNIT.d'
        install -m 644 '$DIR/github-runner-override.conf' '/etc/systemd/system/$GITHUB_UNIT.d/override.conf'
    fi
    systemctl daemon-reload
"

echo "Installed. Verify with:"
echo "  systemctl show gitlab-runner -p Slice"
echo "  systemctl show '$GITHUB_UNIT' -p Slice -p Restart"
echo "Then restart the services when no CI jobs are running."
