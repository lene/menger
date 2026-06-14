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
| TR-9 | FFmpeg/libav runtime mismatch | Video decode failure | CI installs shared libav packages; fail video tests clearly |
| TR-10 | Large video inputs | CPU/GPU memory pressure | 8-frame CPU cache and one stable GPU slot per active source |
| TR-11 | AI review diffs sent to DeepSeek API | MR diff content leaves the local environment | Accepted: this is a public codebase; revisit if proprietary code appears. Diffs are not stored by DeepSeek beyond the API session per their privacy policy. `DEEPSEEK_API_KEY` stored as masked+protected CI variable; never echoed to logs. |
| TR-12 | Single GPU shared between interactive use and CI runners | CI GPU job starves rendering work (or vice versa); flaky integration tests under contention | `limit = 1` serializes menger GPU jobs; GPU-tagged jobs from other pipelines are independent. Known flakes documented in `docs/TESTING.md`. |
| TR-13 | Local CI runner unavailable (host off, OOM, service crash) | All pipelines queue indefinitely; MR merge blocked | systemd `Restart=on-failure` + heartbeat alert job (28.3); nightly alert fires if no job picked up in 24 h. |

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
| TD-7 | ~~Cone and plane shaders lack image texture + PBR map support~~ — resolved Sprint 21.6 | Resolved | — |

### Deferred Features

| Feature | Reason | Status |
|---------|--------|--------|
| Dynamic window resize | Complex timing issues | Backlog |
| Caustics (PPM) | Algorithm produces incorrect results | Branch preserved |
| ~~Mixed geometry scenes~~ | Spheres + any triangle-mesh combination supported via per-instance multi-GAS IAS | Resolved (Sprint 18.1, TD-5) |
| Colored transparent shadows Phase 2 | Phase 1 (closesthit, single-object) complete; Phase 2 anyhit accumulation for overlapping transparent objects remains | TD-6 |
| ~~Cone/plane image + PBR map textures~~ | UV generation + shader sampling implemented for cone and plane | Resolved (Sprint 21.6, TD-7) |

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
| Video texture fails to load | Missing/incompatible libav codec or bad path | Install FFmpeg/libav runtime packages; check `--texture-dir` |
| Env-map video rejected | Video is not equirectangular 2:1 | Use a 360-degree equirectangular source |

## 11.4 Monitoring

### Key Metrics

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| Test count | CI | < 2500 (regression floor; current: 2,823) |
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
