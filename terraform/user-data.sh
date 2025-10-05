#!/bin/bash
# Minimal user-data for instance-specific setup
# AMI should already have CUDA, OptiX, and all dev tools installed

set -e

# Log all output
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "=== Instance initialization at $(date) ==="

# Set fish as default shell for ubuntu user
chsh -s /usr/bin/fish ubuntu

# Setup environment for ubuntu user
sudo -u ubuntu mkdir -p /home/ubuntu/.config/fish
sudo -u ubuntu cp /etc/skel/.config/fish/config.fish /home/ubuntu/.config/fish/config.fish
sudo -u ubuntu cp /etc/skel/.bashrc /home/ubuntu/.bashrc

# Create workspace directory
sudo -u ubuntu mkdir -p /home/ubuntu/workspace

# Clone menger repository
cd /home/ubuntu/workspace
sudo -u ubuntu git clone https://gitlab.com/lilacashes/menger.git

# Create welcome message
cat > /home/ubuntu/WELCOME.txt <<'EOF'
Welcome to your NVIDIA GPU Development Instance!

GPU Information:
  Run: nvidia-smi

CUDA:
  Version: 12.8
  Path: /usr/local/cuda-12.8

OptiX:
  Location: /opt/optix

Development Tools:
  - Scala/sbt: Run 'sbt' in the menger directory
  - IntelliJ IDEA: Run 'intellij-idea-community'
  - Fish shell: Already set as default
  - Git: menger repo cloned to ~/workspace/menger

X11 Forwarding:
  Already configured. Connect with: ssh -X ubuntu@<instance-ip>
  Test with: xclock

Useful Commands:
  - Check GPU: nvidia-smi
  - Build menger: cd ~/workspace/menger && sbt compile
  - Run tests: cd ~/workspace/menger && sbt test

Project directory: ~/workspace/menger
EOF

chown ubuntu:ubuntu /home/ubuntu/WELCOME.txt

echo "=== Initialization complete at $(date) ==="
