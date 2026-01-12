# 10. Quality Requirements

> **Note:** Target values in this document are placeholders to be validated during sprint implementation.
> Each sprint references these scenarios and establishes actual baselines.
>
> **Sprint Traceability:**
> - [Caustics Test Ladder](../caustics/CAUSTICS_TEST_LADDER.md) - Caustics validation (C1-C8)

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

| ID | Scenario | Measure | Target | Status |
|----|----------|---------|--------|--------|
| P1 | Sponge level 3 generation | Time | < 5s | See performance docs |
| P2 | OptiX render 800×600 | Time | < 500ms | See performance docs |
| P3 | OptiX render with shadows | Overhead | < 30% | ✅ Validated |
| P4 | Adaptive AA (depth 2) | Samples vs uniform | 5-20× fewer | ✅ Validated |
| P5 | Statistics collection | Overhead | < 5% | ✅ Validated |

### Visual Quality

| ID | Scenario | Measure | Target | Status |
|----|----------|---------|--------|--------|
| V1 | Glass refraction | IOR accuracy | Matches Snell's law | ✅ Validated |
| V2 | Fresnel reflection | Edge vs center | Correct angular falloff | ✅ Validated |
| V3 | Beer-Lambert absorption | Color tinting | Physically plausible | ✅ Validated |
| V4 | Shadow edges | Sharpness | No acne, no peter-panning | ✅ Validated |
| V5 | Antialiasing | Edge smoothness | No visible jaggies | ✅ Validated |

### Caustics Quality (Progressive Photon Mapping)

> **Validation Approach:** Caustics validation uses a **ladder of tests** from analytic physics
> to reference image comparison. Each step must pass before proceeding to the next.
> See [Caustics Test Ladder](../caustics/CAUSTICS_TEST_LADDER.md) for implementation details.

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

| ID | Scenario | Measure | Target | Status |
|----|----------|---------|--------|--------|
| R1 | All tests pass | Count | 1,070 tests passing | ✅ Current |
| R2 | GPU error recovery | Cache corruption | Auto-recover | ✅ Validated |
| R3 | Invalid CLI args | Error message | Clear, actionable | ✅ Validated |
| R4 | Missing GPU | Graceful degradation | LibGDX fallback | ✅ Existing |

### Maintainability

| ID | Scenario | Measure | Target | Status |
|----|----------|---------|--------|--------|
| M1 | Code style | Scalafix violations | 0 | ✅ Enforced |
| M2 | Wartremover | Warnings | 0 | ✅ Enforced |
| M3 | Line length | Characters | ≤ 100 | ✅ Enforced |
| M4 | New geometry type | Implementation time | < 1 day | ✅ Validated |

## 10.3 Test Coverage

### By Component

| Component | Tests | Coverage Focus |
|-----------|-------|----------------|
| menger.objects | ~50 | Geometry generation, subdivision |
| menger.input | ~20 | Event handling, camera |
| optix-jni (Scala) | ~950 | Renderer API, materials, integration |
| optix-jni (C++) | 27 | OptiXContext primitives |
| CLI | ~30 | Argument parsing, validation |

### Test Types

| Type | Purpose | Run Frequency |
|------|---------|---------------|
| Unit | Logic verification | Every commit |
| Integration | Component interaction | Every commit |
| Visual | Rendering correctness | Manual/PR |
| Performance | Timing benchmarks | Weekly |

## 10.4 Performance Budgets

> **Note:** Actual performance metrics are measured during development and documented in sprint planning.
> For detailed performance data, see [docs/sprints/](../../sprints/) and [docs/archive/sprints/](../archive/sprints/).

### Render Time (800×600)

| Configuration | Target Budget | Status |
|---------------|---------------|--------|
| Base render (sphere) | < 200ms | ✅ Baseline established |
| Base render (cube) | < 300ms | ✅ Baseline established |
| + Shadows | < 30% overhead | ✅ Validated |
| + 4 lights | < 50% overhead | ✅ Validated |
| + AA depth 2 | < 500ms | ✅ Validated |
| + Sponge level 2 | < 1s | ✅ Baseline established |

### Memory (GPU)

| Resource | Typical Usage | Notes |
|----------|---------------|-------|
| Frame buffer | 8 MB (1920×1080×4) | Fixed per resolution |
| GAS (sphere) | < 1 MB | Compact primitive |
| GAS (cube) | < 2 MB | Small triangle mesh |
| GAS (sponge L3) | < 50 MB | Depends on algorithm |
| AA samples buffer | < 10 MB | Adaptive allocation |
| Textures | Varies | User-dependent |

### Performance Baselines

For detailed performance metrics and historical data, refer to:
- **Sprint documentation:** [docs/sprints/](../../sprints/) and [docs/archive/sprints/](../archive/sprints/)
- **Code quality assessments:** [CODE_IMPROVEMENTS.md](../../CODE_IMPROVEMENTS.md)
