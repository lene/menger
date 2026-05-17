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
| TR-6 | LibGDX input processor ordering | UI issues | Separate InputProcessors via OptiXInputMultiplexer |

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
| TD-4 | Caustics: multi-light and multi-plane not yet supported | Low | 4-8h (single light[0] and plane[0] only) |
| TD-5 | ~~Cannot mix spheres with triangle meshes~~ — resolved Sprint 18.1 | Resolved | — |
| TD-6 | Colored transparent shadows Phase 2 (multi-object) | Low | 4-8h (anyhit accumulation for overlapping transparent objects) |
| TD-7 | Cone and plane shaders lack image texture + PBR map support | Medium | 1-2 days per geometry type (UV generation + shader sampling; see CODE_IMPROVEMENTS M-texture-builder-gap) |

### Deferred Features

| Feature | Reason | Status |
|---------|--------|--------|
| Dynamic window resize | Complex timing issues | Backlog |
| Caustics (PPM) | Algorithm produces incorrect results | Branch preserved |
| ~~Mixed geometry scenes~~ | Spheres + any triangle-mesh combination supported via per-instance multi-GAS IAS | Resolved (Sprint 18.1, TD-5) |
| Colored transparent shadows Phase 2 | Phase 1 (closesthit, single-object) complete; Phase 2 anyhit accumulation for overlapping transparent objects remains | TD-6 |
| Cone/plane image + PBR map textures | `texture_index` field repurposed for geometry data; UV coords not generated. Procedural textures work. Image textures and normal/roughness maps deferred. | TD-7 |

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
| Test count | CI | < 1683 (regression floor; current: 1,710) |
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

1. Skip OptiX tests (CI skips if no GPU)
2. Use AWS EC2 spot instance with GPU
3. No software fallback renderer — OptiX is the sole renderer (AD-16)

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
