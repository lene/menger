# 4. Solution Strategy

## 4.1 Technology Decisions

| Decision | Rationale |
|----------|-----------|
| **Scala 3** | Functional programming, immutability, pattern matching, strong type system |
| **LibGDX/LWJGL3** | Cross-platform windowing, input handling (3D rendering removed in AD-16) |
| **OptiX** | Hardware-accelerated ray tracing, physically-based rendering |
| **JNI** | Bridge Scala to native C++/CUDA code |
| **Surface Subdivision** | O(12^n) complexity vs O(20^n) volume-based |

## 4.2 Architecture Approach

### Rendering Pipeline (OptiX-only since v0.5.7 / AD-16)

```
                    ┌─────────────────────┐
                    │   Scala Geometry    │
                    │   (Menger, Cube,    │
                    │    Tesseract, 4D)   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   OptiX Path        │
                    │   (Ray Tracing)     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Triangle Mesh /   │
                    │   IS programs       │
                    │   (GPU Buffers)     │
                    └─────────────────────┘
```

LibGDX/LWJGL3 is retained solely for windowing and input (keyboard,
mouse, orbit camera). All rendering is OptiX ray tracing (see AD-16).

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

### Layered Architecture

The application is organised as an onion of six layers (L0 inward, L5
outward). Dependencies must point inward only — outer layers may import
from inner layers, never the reverse. The full per-package interface
breakdown lives in §5 (Building Block View); this is the strategy-level
overview.

| Layer | Packages | Module | Purpose |
|------:|----------|--------|---------|
| L0 | `menger.common` | `menger-common` | Shared primitive types. The most stable layer. |
| L1 | `menger.objects`, `menger.dsl` | `menger-app` | Pure geometry and the scene-description DSL. |
| L2 | `menger.config` | `menger-app` | Configuration case classes assembled from CLI input and DSL output. |
| L3 | `menger.optix` (wrapper), `menger.input` | `menger-app` | Ports / adapters. The wrapper adapts the JNI module to engines; `menger.input` adapts LibGDX. |
| L4 | `menger.engines` | `menger-app` | Orchestration of the render loop, animation, video export. |
| L5 | `menger.cli`, `Main.scala` | `menger-app` | Application boundary — argument parsing and process entry. |
| ext | `menger.optix.*` (native) | `optix-jni` | External adapter: native OptiX/CUDA bindings. |
| ext | LibGDX, LWJGL3 | external jars | External adapter: windowing and input. |

The layering is enforced at test time by `ArchitectureSpec` and
`ArchitecturePhase2Spec` in `menger-app/src/test/scala/menger/`. Currently
accepted deviations are tracked in `CODE_IMPROVEMENTS.md` under `M-arch-*`
entries (see AD-23 in §9).

## 4.3 Quality Strategy

| Quality Goal | Strategy |
|--------------|----------|
| **Performance** | Surface subdivision, GPU acceleration, BVH optimization |
| **Visual Quality** | Fresnel reflection, Beer-Lambert absorption, adaptive AA |
| **Testability** | 2,823+ tests, separate unit/integration tests, ArchUnit architecture enforcement |
| **Maintainability** | Wartremover, Scalafix enforcement |

## 4.4 Organizational Strategy

- **Single source of truth**: arc42 documentation
- **Sprint-based development**: Detailed plans in `docs/archive/sprints/`
- **CI/CD**: Automated testing on GPU-enabled runner
