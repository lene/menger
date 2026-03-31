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
sudo -u ubuntu git clone --branch "${menger_branch}" https://gitlab.com/lilacashes/menger.git

# Build menger-app and install to ~/bin so it is on PATH
echo "=== Building menger-app (sbt stage) ==="
cd /home/ubuntu/workspace/menger
sudo -u ubuntu bash -c "
  export CUDA_HOME=/usr/local/cuda-12.8
  export OPTIX_ROOT=/opt/optix
  export PATH=\$CUDA_HOME/bin:\$PATH
  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
  cd /home/ubuntu/workspace/menger && sbt stage
"
sudo -u ubuntu mkdir -p /home/ubuntu/bin
sudo -u ubuntu ln -sf \
  /home/ubuntu/workspace/menger/menger-app/target/universal/stage/bin/menger-app \
  /home/ubuntu/bin/menger-app

# Add ~/bin to PATH for Bash
grep -q 'HOME/bin' /home/ubuntu/.bashrc || \
  sudo -u ubuntu bash -c "echo 'export PATH=\$HOME/bin:\$PATH' >> /home/ubuntu/.bashrc"

# Add ~/bin to PATH for Fish
grep -q 'fish_add_path.*bin' /home/ubuntu/.config/fish/config.fish 2>/dev/null || \
  sudo -u ubuntu bash -c "echo 'fish_add_path ~/bin' >> /home/ubuntu/.config/fish/config.fish"

echo "=== menger-app installed to ~/bin/menger-app ==="

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

menger-app:
  Already built and on your PATH.
  Run: menger-app --help
  Example: menger-app --optix --sponge-type cube-sponge --level 3 --save-name out.png

Development:
  Source code:  ~/workspace/menger  (branch: ${menger_branch})
  Rebuild:      cd ~/workspace/menger && sbt stage
  Run tests:    cd ~/workspace/menger && xvfb-run sbt test
  IntelliJ:     intellij-idea-community

X11 Forwarding:
  Already configured. Connect with: ssh -X ubuntu@<instance-ip>
  Test with: xclock
EOF

chown ubuntu:ubuntu /home/ubuntu/WELCOME.txt

echo "=== Initialization complete at $(date) ==="
