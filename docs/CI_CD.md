# CI/CD Configuration

## GitLab Runner Setup for OptiX JNI Tests

`Test:OptiXJni` CI job requires GitLab Runner with GPU + OptiX. Complete guide: `optix-jni/RUNNER_SETUP.md`

### Quick Setup

1. **Install NVIDIA Driver** (580.x+ recommended)
2. **Install Docker Engine** (19.03+)
3. **Install NVIDIA Container Toolkit:**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | pkexec gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  pkexec tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
pkexec apt-get update && pkexec apt-get install -y nvidia-container-toolkit
pkexec nvidia-ctk runtime configure --runtime=docker
pkexec systemctl restart docker
```

4. **Install GitLab Runner:**
```bash
curl -L "https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh" | pkexec bash
pkexec apt-get install gitlab-runner
```

5. **Register Runner** (tag: `nvidia`, executor: docker, image: ubuntu:24.04)
```bash
pkexec gitlab-runner register
```

6. **Configure Runner** - Edit `/etc/gitlab-runner/config.toml`:
```toml
[[runners]]
  name = "nvidia-gpu-runner"
  url = "https://gitlab.com/"
  token = "YOUR_RUNNER_TOKEN"
  executor = "docker"
  tags = ["nvidia"]
  [runners.docker]
    tls_verify = false
    image = "ubuntu:24.04"
    privileged = false
    volumes = ["/cache", "/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro"]
    gpus = "all"
```

**Critical:**
- `gpus = "all"` - GPU access
- Volume mount `libnvoptix.so.1` - OptiX runtime (from driver, not SDK)

7. **Restart:** `pkexec gitlab-runner restart`

### OptiX Library Requirement

OptiX 7.0+ runtime (`libnvoptix.so`) comes from driver, must mount into containers. NVIDIA Container Toolkit doesn't auto-mount OptiX.

**Verify:** `ls -la /usr/lib/x86_64-linux-gnu/libnvoptix.so.1`

If missing: Update driver to 580.x+

## Docker Image Build and Push

```bash
# Set version tag (format: {CUDA}-{OptiX}-{Java}-{sbt})
export VERSION=12.8-9.0-25-1.11.7

# Copy OptiX SDK installer
cp ~/Downloads/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh optix-jni/

# Build
docker build -t registry.gitlab.com/lilacashes/menger/optix-cuda:$VERSION -f optix-jni/Dockerfile optix-jni/

# Tag latest
docker tag registry.gitlab.com/lilacashes/menger/optix-cuda:$VERSION registry.gitlab.com/lilacashes/menger/optix-cuda:latest

# Push
docker login registry.gitlab.com
docker push registry.gitlab.com/lilacashes/menger/optix-cuda:$VERSION
docker push registry.gitlab.com/lilacashes/menger/optix-cuda:latest

# Update .gitlab-ci.yml OPTIX_DOCKER_VERSION
```

**Image layers:**
- Base: `nvidia/cuda:12.8.0-devel-ubuntu24.04` (~9GB)
- Build tools (cmake, g++, ~200MB)
- OptiX SDK 9.0 (~500MB)
- Java 25 LTS (~400MB)
- sbt 1.11.7 (~100MB)

## Testing Docker Images Locally

**CRITICAL:** Always run with your user ID to prevent permission issues.

```bash
# Test compilation (no GPU)
docker run --rm \
  --user $(id -u):$(id -g) \
  -v "$PWD:/workspace" \
  -w /workspace \
  registry.gitlab.com/lilacashes/menger/optix-cuda:12.8-9.0-25-1.11.7 \
  bash -c "sbt 'project optixJni' compile"

# Test with GPU
docker run --rm \
  --user $(id -u):$(id -g) \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -v "$PWD:/workspace" \
  -w /workspace \
  registry.gitlab.com/lilacashes/menger/optix-cuda:12.8-9.0-25-1.11.7 \
  bash -c "sbt 'project optixJni' test"
```

**Key flags:**
- `--user $(id -u):$(id -g)` - **CRITICAL**: Run as your user (not root)
- `-v "$PWD:/workspace"` - Mount as /workspace (NOT /builds/lilacashes/menger)
- `-w /workspace` - Working directory
- `--gpus all` - GPU access (runtime tests only)
- `-e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility` - Mount OptiX/RTX libs

**DON'T:**
- `docker run --rm -v "$PWD:/builds/lilacashes/menger"` ❌ (runs as root)
- Mount at GitLab CI path ❌ (CMake cache conflicts)

**Fix permission issues:**
```bash
pkexec chown -R $USER:$USER optix-jni/target/
rm -rf optix-jni/target/native
```

## Troubleshooting CI Failures

### "OPTIX_ERROR_LIBRARY_NOT_FOUND" or "Error initializing RTX library"

**Cause:** OptiX/RTX runtime libs (`libnvoptix.so`, `libnvidia-rtcore.so`) not accessible

**Solution:**

1. **In `.gitlab-ci.yml`:**
```yaml
Test:OptiXJni:
  variables:
    NVIDIA_DRIVER_CAPABILITIES: "graphics,compute,utility"  # Critical!
  before_script:
    - |
      if [ -f /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.* ] && [ ! -f /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.1 ]; then
        ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.* /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.1
      fi
    - ldconfig || true
```

2. **In runner config:** `gpus = "all"`

3. **Verify on host:**
```bash
ls -la /usr/lib/x86_64-linux-gnu/libnvoptix.so*
ls -la /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so*
```

**Background:** `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility` tells NVIDIA Container Toolkit to mount OptiX/RTX libs.

### "could not select device driver with capabilities: [[gpu]]"

**Cause:** NVIDIA Container Toolkit not configured

**Fix:**
```bash
pkexec nvidia-ctk runtime configure --runtime=docker
pkexec systemctl restart docker
pkexec gitlab-runner restart
```

### Runner not picking up 'nvidia' tag jobs

**Cause:** Runner not tagged

**Fix:**
- GitLab UI: Settings > CI/CD > Runners, add 'nvidia' tag
- Or edit `/etc/gitlab-runner/config.toml`: `tags = ["nvidia"]`
- Restart runner

### Job runs on wrong runner

**Cause:** Multiple runners, wrong priority

**Fix:**
```bash
# Pause unwanted runners
glab api --method PUT projects/lilacashes%2Fmenger/runners/<RUNNER_ID> -f "paused=true"
```

### Docker fails after adding "optix" to supported-driver-capabilities

**Issue:** `"optix"` in `/etc/nvidia-container-runtime/config.toml` causes "unsupported capabilities" error

**Root cause:** `optix` capability not universally recognized

**Solution:** DON'T add `optix` to supported-driver-capabilities. Use `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility` environment variable.

### CMake cache conflicts local/Docker

**Cause:** CMake caches absolute paths, invalid when mounted at different path

**Fix:** `rm -rf optix-jni/target/native` before Docker tests

(GitLab CI auto-cleans workspace)

## CUDA Architecture Compatibility

**OptiX JNI:** CUDA architecture **sm_75** (RTX 20 series). Forward-compatible via PTX JIT compilation.

| GPU | Compute Capability | sm_75 PTX Compatible |
|-----|-------------------|----------------------|
| RTX 2080 Ti | sm_75 | ✓ Native |
| RTX 3090 | sm_86 | ✓ JIT |
| RTX 4090 | sm_89 | ✓ JIT |
| RTX A1000 | sm_86 | ✓ JIT |
| Tesla T4 | sm_75 | ✓ Native |

PTX (Parallel Thread Execution) = NVIDIA virtual assembly, ensures forward compatibility.
