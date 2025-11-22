# 4. Solution Strategy

## 4.1 Technology Decisions

| Decision | Rationale |
|----------|-----------|
| **Scala 3** | Functional programming, immutability, pattern matching, strong type system |
| **LibGDX** | Cross-platform OpenGL abstraction, proven game engine |
| **OptiX** | Hardware-accelerated ray tracing, physically-based rendering |
| **JNI** | Bridge Scala to native C++/CUDA code |
| **Surface Subdivision** | O(12^n) complexity vs O(20^n) volume-based |

## 4.2 Architecture Approach

### Dual Rendering Pipeline

```
                    ┌─────────────────────┐
                    │   Scala Geometry    │
                    │   (Menger, Cube,    │
                    │    Tesseract)       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
    ┌─────────────────────┐           ┌─────────────────────┐
    │   LibGDX Path       │           │   OptiX Path        │
    │   (Rasterization)   │           │   (Ray Tracing)     │
    └─────────────────────┘           └─────────────────────┘
              │                                 │
              ▼                                 ▼
    ┌─────────────────────┐           ┌─────────────────────┐
    │   ModelInstance     │           │   Triangle Mesh     │
    │   (OpenGL)          │           │   (GPU Buffers)     │
    └─────────────────────┘           └─────────────────────┘
```

### Key Design Principles

1. **Separation of Concerns**
   - Geometry generation independent of rendering
   - OptiXContext (low-level) vs OptiXWrapper (high-level)

2. **Functional Style**
   - Immutable data structures
   - `Try`/`Either` for error handling
   - Observer pattern for event propagation

3. **Performance Optimization**
   - Scene data in Params (not SBT) for GPU performance
   - Lazy evaluation for expensive geometry
   - Sponge mesh caching per level

## 4.3 Quality Strategy

| Quality Goal | Strategy |
|--------------|----------|
| **Performance** | Surface subdivision, GPU acceleration, BVH optimization |
| **Visual Quality** | Fresnel reflection, Beer-Lambert absorption, adaptive AA |
| **Testability** | 897+ tests, separate unit/integration tests |
| **Maintainability** | Wartremover, Scalafix enforcement |

## 4.4 Organizational Strategy

- **Single source of truth**: arc42 documentation
- **Sprint-based development**: Detailed plans in `optix-jni/SPRINT_*_PLAN.md`
- **CI/CD**: Automated testing on GPU-enabled runner
