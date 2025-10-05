# Packer template for building custom NVIDIA development AMI
# Usage: packer build -var 'optix_installer=path/to/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh' \
#        ami-build.pkr.hcl

packer {
  required_plugins {
    amazon = {
      version = ">= 1.0.0"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "instance_type" {
  type    = string
  default = "g4dn.xlarge"
}

variable "optix_installer" {
  type        = string
  description = "Path to OptiX installer (download from developer.nvidia.com/optix)"
}

source "amazon-ebs" "nvidia_dev" {
  ami_name      = "menger-nvidia-dev-{{timestamp}}"
  instance_type = var.instance_type
  region        = var.region

  source_ami_filter {
    filters = {
      name                = "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    most_recent = true
    owners      = ["099720109477"] # Canonical
  }

  ssh_username = "ubuntu"

  launch_block_device_mappings {
    device_name = "/dev/sda1"
    volume_size = 100
    volume_type = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name        = "menger-nvidia-dev"
    Project     = "menger"
    CUDAVersion = "12.8"
    Built       = "{{timestamp}}"
  }
}

build {
  sources = ["source.amazon-ebs.nvidia_dev"]

  # Upload OptiX installer
  provisioner "file" {
    source      = var.optix_installer
    destination = "/tmp/optix-installer.sh"
  }

  # Main provisioning script
  provisioner "shell" {
    inline = [
      "set -e",
      "sudo apt-get update",
      "sudo apt-get upgrade -y",

      # Basic tools
      "sudo apt-get install -y build-essential git curl wget vim htop tmux unzip jq software-properties-common",

      # Fish shell
      "sudo apt-add-repository ppa:fish-shell/release-3 -y",
      "sudo apt-get update",
      "sudo apt-get install -y fish",

      # X11 support
      "sudo apt-get install -y xauth x11-apps libx11-dev libxext-dev libxrender-dev libxtst-dev libxi-dev",

      # Configure X11 forwarding in SSH
      "sudo sed -i 's/#X11Forwarding yes/X11Forwarding yes/' /etc/ssh/sshd_config",
      "sudo sed -i 's/#X11DisplayOffset 10/X11DisplayOffset 10/' /etc/ssh/sshd_config",
      "sudo sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/' /etc/ssh/sshd_config",

      # Install NVIDIA drivers
      "sudo apt-get install -y ubuntu-drivers-common",
      "sudo ubuntu-drivers install",

      # Install CUDA 12.8
      "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
      "sudo dpkg -i cuda-keyring_1.1-1_all.deb",
      "sudo apt-get update",
      "sudo apt-get install -y cuda-toolkit-12-8",
      "rm cuda-keyring_1.1-1_all.deb",

      # Install OptiX
      "sudo mkdir -p /opt/optix",
      "chmod +x /tmp/optix-installer.sh",
      "sudo /tmp/optix-installer.sh --skip-license --prefix=/opt/optix",
      "rm /tmp/optix-installer.sh",

      # Set environment variables in skeleton files (for new users)
      "echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' | sudo tee -a /etc/skel/.bashrc",
      "echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/skel/.bashrc",
      "echo 'export OPTIX_ROOT=/opt/optix' | sudo tee -a /etc/skel/.bashrc",
      "sudo mkdir -p /etc/skel/.config/fish",
      "echo 'set -x PATH /usr/local/cuda-12.8/bin $PATH' | sudo tee /etc/skel/.config/fish/config.fish",
      "echo 'set -x LD_LIBRARY_PATH /usr/local/cuda-12.8/lib64 $LD_LIBRARY_PATH' | sudo tee -a /etc/skel/.config/fish/config.fish",
      "echo 'set -x OPTIX_ROOT /opt/optix' | sudo tee -a /etc/skel/.config/fish/config.fish",

      # Java for Scala/sbt
      "sudo apt-get install -y openjdk-17-jdk",

      # Install sbt
      "echo 'deb https://repo.scala-sbt.org/scalasbt/debian all main' | sudo tee /etc/apt/sources.list.d/sbt.list",
      "curl -sL 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823' | sudo apt-key add",
      "sudo apt-get update",
      "sudo apt-get install -y sbt",

      # Install Coursier
      "curl -fL https://github.com/coursier/launchers/raw/master/cs-x86_64-pc-linux.gz | gzip -d | sudo tee /usr/local/bin/cs > /dev/null",
      "sudo chmod +x /usr/local/bin/cs",

      # Install IntelliJ IDEA
      "sudo snap install intellij-idea-community --classic",

      # Clean up
      "sudo apt-get autoremove -y",
      "sudo apt-get clean",
      "sudo rm -rf /var/lib/apt/lists/*"
    ]
  }
}
