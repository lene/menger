# 11. Risks and Technical Debt

## 11.1 Technical Risks

### High Risk

| ID | Risk | Impact | Mitigation |
|----|------|--------|------------|
| TR-1 | OptiX SDK/driver version mismatch | Crashes (CUDA error 718) | Document compatible versions; CI validates |
| TR-2 | GPU memory exhaustion | Render failure | Limit sponge level; monitor memory |
| TR-3 | PTX file not found after clean build | Startup failure | Auto-copy in sbt; document workaround |

### Medium Risk

| ID | Risk | Impact | Mitigation |
|----|------|--------|------------|
| TR-4 | Large sponge BVH build time | Poor UX | Cache geometry; async build |
| TR-5 | JNI memory leaks | Gradual OOM | RAII patterns; dispose() enforcement |
| TR-6 | LibGDX/OptiX input conflict | UI issues | Separate InputProcessors |

### Low Risk

| ID | Risk | Impact | Mitigation |
|----|------|--------|------------|
| TR-7 | 4D projection edge cases | Visual artifacts | Comprehensive tests |
| TR-8 | Color format confusion | Wrong colors | Unified Color API |

## 11.2 Technical Debt

### Known Debt

| ID | Description | Severity | Effort to Fix |
|----|-------------|----------|---------------|
| TD-1 | Single PTX file growing large | Low | 2-4h (split into modules) |
| TD-2 | Hardcoded MAX_LIGHTS = 8 | Low | 1h (make configurable) |
| TD-3 | Window resize not working | Medium | 15+h (deferred) |
| TD-4 | Caustics algorithm incomplete | High | 20+h (deferred) |

### Deferred Features

| Feature | Reason | Status |
|---------|--------|--------|
| Dynamic window resize | Complex timing issues | Backlog |
| Caustics (PPM) | Algorithm produces incorrect results | Branch preserved |
| Soft shadows | Requires area lights | Future sprint |

## 11.3 Common Issues and Solutions

Detailed troubleshooting: [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)

### Quick Reference

| Symptom | Cause | Solution |
|---------|-------|----------|
| CUDA error 718 | SDK/driver mismatch | Install matching OptiX SDK |
| PTX not found | Clean build | Copy from optix-jni/target to target |
| Permission errors | Docker builds | `chown -R $USER:$USER optix-jni/target/` |
| Cache corruption | OptiX crash | Delete `~/.cache/nvidia-optix-cache` |
| Gimbal lock | 90° elevation | Clamp to ±89° |
| libnvidia-glcore crash | Threading | Set `__GL_THREADED_OPTIMIZATIONS=0` |

## 11.4 Monitoring

### Key Metrics

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| Test count | CI | < 897 (regression) |
| Build time | CI | > 10 minutes |
| Render time | RayStats | > 2× baseline |
| GPU memory | nvidia-smi | > 80% |

### Health Checks

```bash
# Verify driver
nvidia-smi

# Verify CUDA
nvcc --version

# Verify OptiX
strings /usr/lib/x86_64-linux-gnu/libnvoptix.so.* | grep "OptiX Version"

# Verify tests
sbt test --warn
```

## 11.5 Contingency Plans

### GPU Unavailable

1. Fall back to LibGDX rendering (automatic)
2. Skip OptiX tests (CI skips if no GPU)
3. Use AWS EC2 spot instance

### CI Failure

1. Check runner health (nvidia-smi)
2. Restart GitLab runner service
3. Clear Docker cache
4. Re-register runner

### Corrupted Cache

```bash
# Clear OptiX cache
rm -rf ~/.cache/nvidia-optix-cache

# Clean rebuild
rm -rf optix-jni/target/native
sbt "project optixJni" compile
```
