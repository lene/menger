# 10. Quality Requirements

> **Note:** Target values in this document are placeholders to be validated during sprint implementation.
> Each sprint references these scenarios and establishes actual baselines.
>
> **Sprint Traceability:**
> - [Sprint 4](../../optix-jni/CAUSTICS_TEST_LADDER.md) - Caustics validation (C1-C8)
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

### Caustics Quality (Progressive Photon Mapping)

> **Validation Approach:** Caustics validation uses a **ladder of tests** from analytic physics
> to reference image comparison. Each step must pass before proceeding to the next.
> See [Caustics Test Ladder](../../optix-jni/CAUSTICS_TEST_LADDER.md) for implementation details.

| ID | Scenario | Measure | Target | Test Type |
|----|----------|---------|--------|-----------|
| C1 | Photon emission | Count & direction | Matches requested, uniform distribution | Analytic |
| C2 | Sphere hit rate | Photons hitting sphere | Within 5% of geometric cross-section | Analytic |
| C3 | Refraction accuracy | Exit angle vs Snell's law | < 0.01 rad error for known angles | Analytic |
| C4 | Focal point position | Caustic centroid location | Within 0.1 units of predicted focal point | Analytic |
| C5 | Energy conservation | Flux in vs flux deposited | Within 5% (accounting for absorption) | Statistical |
| C6 | PPM convergence | Variance over iterations | Monotonic decrease after iteration 3 | Statistical |
| C7 | Caustic brightness | Peak vs ambient | > 1.5× brighter at focal point | Statistical |
| C8 | Reference match | SSIM to known-good render | > 0.90 for canonical scene | Image comparison |

#### Caustics Test Ladder

```
Step 8: Reference Match ──────── C8: SSIM > 0.90
    ▲
Step 7: Brightness ───────────── C7: Peak > 1.5× ambient
    ▲
Step 6: Convergence ──────────── C6: Variance decreases
    ▲
Step 5: Energy Conservation ──── C5: Flux in ≈ flux out
    ▲
Step 4: Focal Point ──────────── C4: Centroid at predicted position
    ▲
Step 3: Refraction ───────────── C3: Exit angles match Snell's law
    ▲
Step 2: Sphere Hits ──────────── C2: Hit rate matches geometry
    ▲
Step 1: Emission ─────────────── C1: Correct count & distribution
```

#### Canonical Test Scene

For reproducible validation, use this fixed configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sphere center | (0, 0, 0) | Origin for simple geometry |
| Sphere radius | 1.0 | Unit sphere |
| Sphere IOR | 1.5 | Standard glass |
| Plane axis | Y | Horizontal floor |
| Plane position | -2.0 | Below sphere, within focal range |
| Light direction | (0, -1, 0) | Vertical, creates circular caustic |
| Camera position | (0, 1, 4) | Above and behind, sees caustic |

**Expected caustic:** Circular bright spot centered at (0, -2, 0) with radius ~0.3 units.

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
