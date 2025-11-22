# 10. Quality Requirements

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

| ID | Scenario | Measure | Target |
|----|----------|---------|--------|
| P1 | Sponge level 3 generation | Time | < 5 seconds |
| P2 | OptiX render 800×600 | Time | < 500ms |
| P3 | OptiX render with shadows | Overhead | < 30% |
| P4 | Adaptive AA (depth 2) | Samples vs uniform | 5-20× fewer |
| P5 | Statistics collection | Overhead | < 5% |

### Visual Quality

| ID | Scenario | Measure | Target |
|----|----------|---------|--------|
| V1 | Glass refraction | IOR accuracy | Matches Snell's law |
| V2 | Fresnel reflection | Edge vs center | Correct angular falloff |
| V3 | Beer-Lambert absorption | Color tinting | Physically plausible |
| V4 | Shadow edges | Sharpness | No acne, no peter-panning |
| V5 | Antialiasing | Edge smoothness | No visible jaggies |

### Reliability

| ID | Scenario | Measure | Target |
|----|----------|---------|--------|
| R1 | All tests pass | Count | 897+ tests passing |
| R2 | GPU error recovery | Cache corruption | Auto-recover |
| R3 | Invalid CLI args | Error message | Clear, actionable |
| R4 | Missing GPU | Graceful degradation | LibGDX fallback |

### Maintainability

| ID | Scenario | Measure | Target |
|----|----------|---------|--------|
| M1 | Code style | Scalafix violations | 0 |
| M2 | Wartremover | Warnings | 0 |
| M3 | Line length | Characters | ≤ 100 |
| M4 | New geometry type | Implementation time | < 1 day |

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

### Render Time (800×600)

| Configuration | Budget |
|---------------|--------|
| Base render | 150ms |
| + Shadows | 200ms |
| + 4 lights | 350ms |
| + AA depth 2 | 800ms |
| + Sponge level 2 | 3s |

### Memory (GPU)

| Resource | Budget |
|----------|--------|
| Frame buffer | 8 MB (1920×1080×4) |
| GAS (sphere) | 1 MB |
| GAS (sponge L3) | 50 MB |
| AA samples | 100 MB max |
| Textures | 256 MB max |
