# 10. Quality Requirements

> **Note:** Target values in this document are placeholders to be validated during sprint implementation.
> Each sprint references these scenarios and establishes actual baselines.
>
> **Sprint Traceability:**
> - [Sprint 5](../../optix-jni/SPRINT_5_PLAN.md) - Establishes cube/triangle baselines
> - [Sprint 6](../../optix-jni/SPRINT_6_PLAN.md) - Establishes sponge/multi-object baselines
> - [Sprint 7](../../optix-jni/SPRINT_7_PLAN.md) - Validates material/texture performance

## 10.1 Quality Tree

```
                    ┌─────────────────────────────────┐
                    │         Quality Goals           │
                    └────────────────┬────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  Performance  │          │    Quality    │          │  Reliability  │
└───────┬───────┘          └───────┬───────┘          └───────┬───────┘
        │                          │                          │
   ┌────┴────┐                ┌────┴────┐                ┌────┴────┐
   ▼         ▼                ▼         ▼                ▼         ▼
Render   Geometry         Visual    Physical         Test      Error
Speed    Efficiency      Quality    Accuracy       Coverage   Handling
```

## 10.2 Quality Scenarios

### Performance

| ID | Scenario | Measure | Target | Validated By |
|----|----------|---------|--------|--------------|
| P1 | Sponge level 3 generation | Time | TBD (target < 5s) | Sprint 6 |
| P2 | OptiX render 800×600 | Time | TBD (target < 500ms) | Sprint 5, 6 |
| P3 | OptiX render with shadows | Overhead | TBD (target < 30%) | Sprint 1 ✅ |
| P4 | Adaptive AA (depth 2) | Samples vs uniform | 5-20× fewer | Sprint 3 ✅ |
| P5 | Statistics collection | Overhead | < 5% | Sprint 1 ✅ |

### Visual Quality

| ID | Scenario | Measure | Target | Validated By |
|----|----------|---------|--------|--------------|
| V1 | Glass refraction | IOR accuracy | Matches Snell's law | Sprint 5, 7 |
| V2 | Fresnel reflection | Edge vs center | Correct angular falloff | Sprint 5, 7 |
| V3 | Beer-Lambert absorption | Color tinting | Physically plausible | Sprint 5, 7 |
| V4 | Shadow edges | Sharpness | No acne, no peter-panning | Sprint 1 ✅, 5 |
| V5 | Antialiasing | Edge smoothness | No visible jaggies | Sprint 3 ✅ |

### Reliability

| ID | Scenario | Measure | Target | Validated By |
|----|----------|---------|--------|--------------|
| R1 | All tests pass | Count | 897+ tests passing | All sprints |
| R2 | GPU error recovery | Cache corruption | Auto-recover | Sprint 3 ✅ |
| R3 | Invalid CLI args | Error message | Clear, actionable | Sprint 5 |
| R4 | Missing GPU | Graceful degradation | LibGDX fallback | Existing |

### Maintainability

| ID | Scenario | Measure | Target | Validated By |
|----|----------|---------|--------|--------------|
| M1 | Code style | Scalafix violations | 0 | All sprints |
| M2 | Wartremover | Warnings | 0 | All sprints |
| M3 | Line length | Characters | ≤ 100 | All sprints |
| M4 | New geometry type | Implementation time | TBD (target < 1 day) | Sprint 5 |

## 10.3 Test Coverage

### By Component

| Component | Tests | Coverage Focus |
|-----------|-------|----------------|
| menger.objects | ~50 | Geometry generation, subdivision |
| menger.input | ~20 | Event handling, camera |
| optix-jni (Scala) | ~100 | Renderer API, materials |
| optix-jni (C++) | 21 | OptiXContext primitives |
| CLI | ~30 | Argument parsing, validation |

### Test Types

| Type | Purpose | Run Frequency |
|------|---------|---------------|
| Unit | Logic verification | Every commit |
| Integration | Component interaction | Every commit |
| Visual | Rendering correctness | Manual/PR |
| Performance | Timing benchmarks | Weekly |

## 10.4 Performance Budgets

> **Note:** Values below are initial estimates. Actual baselines will be established by sprints and documented here.

### Render Time (800×600)

| Configuration | Budget | Status |
|---------------|--------|--------|
| Base render (sphere) | TBD | Sprint 1-3 baseline |
| Base render (cube) | TBD | Sprint 5 to establish |
| + Shadows | TBD | Sprint 1 measured |
| + 4 lights | TBD | Sprint 2 measured |
| + AA depth 2 | TBD | Sprint 3 measured |
| + Sponge level 2 | TBD | Sprint 6 to establish |

### Memory (GPU)

| Resource | Budget | Status |
|----------|--------|--------|
| Frame buffer | 8 MB (1920×1080×4) | Fixed calculation |
| GAS (sphere) | TBD | Sprint 5 to measure |
| GAS (cube) | TBD | Sprint 5 to establish |
| GAS (sponge L3) | TBD | Sprint 6 to establish |
| AA samples | TBD | Sprint 3 measured |
| Textures | TBD | Sprint 7 to establish |

### Baseline Values (Updated by Sprints)

*This section will be populated as sprints complete and establish actual metrics.*

| Sprint | Metric | Value | Date |
|--------|--------|-------|------|
| 1-3 | Sphere render 800×600 | ~150ms | Completed |
| 5 | Cube render 800×600 | TBD | - |
| 6 | Sponge L3 generation | TBD | - |
| 6 | Sponge L3 GAS build | TBD | - |
| 7 | Texture upload (1024×1024) | TBD | - |
