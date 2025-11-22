# 7. Deployment View

## 7.1 Infrastructure Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Development Environment                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────┐        ┌────────────────────────────────┐   │
│  │  Local Workstation │        │  AWS EC2 Spot Instance         │   │
│  │  (with NVIDIA GPU) │   OR   │  (g4dn.xlarge / p3.2xlarge)    │   │
│  └────────────────────┘        └────────────────────────────────┘   │
│                                                                      │
│  Required Software:                                                  │
│  - JVM 21+                                                          │
│  - sbt 1.x                                                          │
│  - NVIDIA Driver 580.x+                                             │
│  - CUDA Toolkit 12.0+                                               │
│  - OptiX SDK 9.0+                                                   │
│  - CMake 3.18+                                                      │
│  - C++17 compiler (gcc/clang)                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          CI/CD Pipeline                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    GitLab Runner (nvidia tag)                 │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │   │
│  │  │ Build Stage    │  │ Test Stage     │  │ Quality Stage  │  │   │
│  │  │ - sbt compile  │  │ - sbt test     │  │ - scalafix     │  │   │
│  │  │ - native build │  │ - GPU tests    │  │ - wartremover  │  │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Runner Requirements:                                                │
│  - Docker with nvidia-container-toolkit                             │
│  - GPU passthrough (gpus = "all")                                   │
│  - OptiX runtime mount (/usr/lib/x86_64-linux-gnu/libnvoptix.so.1) │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 7.2 Local Development Setup

### Automated Setup (Recommended)

```bash
# Install all dependencies
./scripts/setup-dev-environment.sh

# Verify installation
sbt compile && sbt test --warn
```

### Manual Setup

1. **NVIDIA Driver** (580.x+ for OptiX 9.0)
   ```bash
   ubuntu-drivers devices
   sudo ubuntu-drivers autoinstall
   ```

2. **CUDA Toolkit 12.x**
   ```bash
   # Download from NVIDIA
   sudo sh cuda_12.x.x_xxx.xx_linux.run
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

3. **OptiX SDK 9.0+**
   ```bash
   # Download from NVIDIA developer portal
   # Extract to /opt/optix or $HOME/optix
   export OPTIX_PATH=/opt/optix
   ```

4. **Verify**
   ```bash
   nvidia-smi                        # Driver version
   nvcc --version                    # CUDA version
   strings /usr/lib/x86_64-linux-gnu/libnvoptix.so.* | grep "OptiX Version"
   ```

## 7.3 AWS Cloud Development

For developers without local NVIDIA GPU, use AWS EC2 spot instances.

### Instance Types

| Type | GPU | vCPU | RAM | Spot Price |
|------|-----|------|-----|------------|
| g4dn.xlarge | T4 (16GB) | 4 | 16GB | ~$0.16/hr |
| g4dn.2xlarge | T4 (16GB) | 8 | 32GB | ~$0.24/hr |
| p3.2xlarge | V100 (16GB) | 8 | 61GB | ~$0.92/hr |

### Quick Start

```bash
# Launch instance (creates if needed)
./scripts/aws-gpu-instance.sh launch

# Connect with X11 forwarding
./scripts/aws-gpu-instance.sh connect

# Work on instance...

# Save state and stop
./scripts/aws-gpu-instance.sh stop

# Terminate when done
./scripts/aws-gpu-instance.sh terminate
```

### Pre-configured AMI

Custom AMI includes:
- CUDA 12.8
- OptiX SDK 9.0
- JVM 21, sbt, Scala
- IntelliJ IDEA
- Fish shell

## 7.4 CI/CD Configuration

### GitLab Runner Setup

1. **Install NVIDIA Container Toolkit**
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
     sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

2. **Register Runner**
   ```bash
   sudo gitlab-runner register \
     --url "https://gitlab.com/" \
     --registration-token "YOUR_TOKEN" \
     --executor "docker" \
     --docker-image "ubuntu:24.04" \
     --tag-list "nvidia"
   ```

3. **Configure `/etc/gitlab-runner/config.toml`**
   ```toml
   [[runners]]
     name = "nvidia-gpu-runner"
     executor = "docker"
     [runners.docker]
       image = "ubuntu:24.04"
       gpus = "all"
       volumes = [
         "/cache",
         "/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro"
       ]
   ```

## 7.5 Build Artifacts

```
target/
├── native/x86_64-linux/bin/
│   ├── liboptix_jni.so          # Native library
│   └── sphere_combined.ptx       # Compiled CUDA shaders
└── scala-3.x/classes/
    └── native/x86_64-linux/
        └── sphere_combined.ptx   # Copied for classpath loading
```

**Note:** PTX files must be accessible at runtime. sbt build copies to classpath.
