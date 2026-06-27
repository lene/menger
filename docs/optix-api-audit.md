# OptiX API Audit — Sprint 30

**Date:** 2026-06-27
**Status:** optix-jni 0.1.2, OptiX SDK 9.0
**Purpose:** Authoritative 1.0 scope definition — 1.0 means "everything we chose to expose is stable,"
not "everything is wrapped."

## Summary

optix-jni exposes a focused subset of the OptiX 9.0 API — enough to build a production ray tracer with
triangles, curves, spheres, multi-object IAS scenes, AI denoising, and texture management. Of 22 feature
groups in the OptiX 9.0 API, 11 are exposed, 2 are planned for Sprint 30, and 9 are deliberately deferred
past 1.0 with rationale.

---

## Feature Audit Table

### Acceleration Structures

| Feature | Status | Rationale | Target |
|---------|--------|-----------|--------|
| Triangle GAS (built-in) | ✅ Exposed | Core primitive. `OptiXMeshApi.setTriangleMesh` | — |
| Instance AS (IAS) | ✅ Exposed | Multi-object scenes. `addSphereInstance`, `addTriangleMeshInstance`, etc. | — |
| Multi-GAS + recursive IAS | ✅ Exposed | `addRecursiveIASSpongeInstance` — O(level·20) VRAM sponge | — |
| AS compaction | ❌ Deferred | OptiX can compact built AS to reduce memory. Minor VRAM savings; not needed for current scene sizes. | — |
| AS updates (refit) | ✅ Exposed | `updateCpuTriangleMesh`, `updateMesh4DProjection` — GAS refit for animation | — |
| AS relocation | ❌ Deferred | OptiX can pool AS memory. Optimization only; no API impact. | — |
| Motion blur (matrix) | ❌ Deferred | Production rendering feature. Requires per-vertex motion vectors + time-interpolated transforms. Sprint 30 rescoped it out. | Sprint 31+ |
| Motion blur (SRT) | ❌ Deferred | Alternate motion representation. Same rationale as matrix motion blur. | Sprint 31+ |
| Opacity micromaps (OMM) | ❌ Deferred | Alpha-tested geometry acceleration. Requires micro-triangle/micro-mesh infrastructure not yet built. Niche. | Post-1.0 |
| Displaced micro-meshes (DMM) | ❌ Deferred | Tessellated displacement. Major feature with significant API surface. | Post-1.0 |
| Clusters | ❌ Deferred | New OptiX 9.0 primitive for large-scale geometry. Not yet widely adopted. | Post-1.0 |

### Geometry Primitives

| Feature | Status | Rationale | Target |
|---------|--------|-----------|--------|
| Built-in triangle intersection | ✅ Exposed | Via `createTriangleHitGroup` + triangle GAS | — |
| Custom sphere intersection | ✅ Exposed | `hit_sphere.cu` — custom IS program for analytical spheres | — |
| Custom cylinder intersection | ✅ Exposed | `hit_cylinder.cu` | — |
| Custom cone intersection | ✅ Exposed | `hit_cone.cu` | — |
| Custom plane intersection | ✅ Exposed | `miss_plane.cu` + legacy plane API | — |
| Built-in curve (cubic B-spline) | ✅ Exposed | Sprint 29. `createCurveHitGroup` + `addCurveInstance` | — |
| Built-in curve (linear) | ❌ Planned | Low effort — alternate curve type. | Sprint 31 |
| Built-in curve (quadratic) | ❌ Planned | Low effort — alternate curve type. | Sprint 31 |
| Built-in curve (Catmull-Rom) | ❌ Planned | Low effort — alternate curve type. | Sprint 31 |
| Built-in curve (ribbons) | ❌ Deferred | Camera-facing flat curves. Niche; no current use case. | Post-1.0 |
| Built-in sphere primitive | ❌ Deferred | OptiX built-in sphere intersection. Our custom IS is already working and faster for analytical scenes. Switch would require SBT rework. | Post-1.0 |
| Curve back-face culling | ❌ Deferred | OptiX supports per-curve culling flags. Not exposed in current API. | Post-1.0 |

### Instancing

| Feature | Status | Rationale | Target |
|---------|--------|-----------|--------|
| 4x3 affine transform instances | ✅ Exposed | Standard instancing via `add*Instance` methods | — |
| Instance visibility masks | ❌ Deferred | OptiX `instanceMask` — enables selective ray tracing per-instance. Useful for LOD or layer toggles. | Post-1.0 |
| Instance SBT offset | ❌ Deferred | Per-instance shader binding table offset. Not needed with current hitgroup-per-primitive architecture. | Post-1.0 |
| Dynamic instance transforms | ❌ Deferred | Requires AS update + IAS rebuild. Our 4D update path already does this for projection changes. | Post-1.0 |
| Instance random access (device) | ❌ Deferred | `optixGetInstanceIdFromHandle` — needed for complex instancing hierarchies. | Post-1.0 |
| Multi-level transform lists | ❌ Deferred | `optixGetTransformListHandle` — matrix/SRT chains. Overkill for current scenes. | Post-1.0 |

### Shader Programs

| Feature | Status | Rationale | Target |
|---------|--------|-----------|--------|
| Ray generation (RG) | ✅ Exposed | `createRaygenGroup` + `raygen_primary.cu` | — |
| Closest-hit (CH) | ✅ Exposed | `createHitGroup` — per-primitive hit shaders | — |
| Intersection (IS) | ✅ Exposed | Custom IS programs for sphere, cylinder, cone, 4D fractals | — |
| Miss (MS) | ✅ Exposed | `createMissGroup` + `miss_plane.cu` | — |
| Any-hit (AH) | ⚠️ Internal | Used internally for shadow rays. Not exposed as a public API target. | Post-1.0 |
| Exception (EX) | ❌ Deferred | OptiX exception programs for error handling in shader code. Currently errors manifest as silent misses. | Post-1.0 |
| Direct callable (DC) | ❌ Deferred | Callable shaders for code reuse. No current use case. | Post-1.0 |
| Continuation callable (CC) | ❌ Deferred | Continuation-passing callables. Niche. | Post-1.0 |

### AI Denoiser

| Feature | Status | Rationale | Target |
|---------|--------|-----------|--------|
| HDR denoiser (color-only) | ✅ Exposed | Sprint 29. `OptiXDenoiser` + `denoiseFloat4` | — |
| HDR denoiser (albedo guide) | ✅ Exposed | `OptiXDenoiser(guideAlbedo=true)` + `DenoiseGuides.albedo()` | — |
| HDR denoiser (normal guide) | ✅ Exposed | `OptiXDenoiser(guideNormal=true)` + `DenoiseGuides.normal()` | — |
| HDR denoiser (both guides) | ✅ Exposed | Best quality. `DenoiseGuides.albedoAndNormal()` | — |
| Temporal denoising mode | ❌ Deferred | OptiX temporal denoiser accumulates across frames. Requires per-frame state management architecture we don't have. | Post-1.0 |
| AOV denoising | ❌ Deferred | Arbitrary output variable denoising. Needs multi-AOV render pipeline. | Post-1.0 |
| Upscaling denoiser | ❌ Deferred | OptiX denoiser can upscale (e.g., 1080p → 4K). Interesting but separate from core rendering path. | Post-1.0 |

### Shader Execution Reordering

| Feature | Status | Rationale | Target |
|---------|--------|-----------|--------|
| `optixReorder` | 🔄 Planned | Task 30.4. Coherence optimization for divergent rays on Ada+ GPUs. | Sprint 30 |

### Validation

| Feature | Status | Rationale | Target |
|---------|--------|-----------|--------|
| Validation mode (`OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL`) | 🔄 Planned | Task 30.4. Catches SBT and payload errors at call site instead of CUDA error 718 at launch. | Sprint 30 |

### Payload & SBT

| Feature | Status | Rationale | Target |
|---------|--------|-----------|--------|
| Single payload semantic | ✅ Exposed | Fixed-structure ray payload via `OptiXData.h` structs | — |
| SBT record management | ✅ Exposed | Managed internally by `OptiXWrapper` | — |
| Multiple payload semantics | ❌ Deferred | OptiX supports multiple payload types. Not needed with current single-payload pipeline. | Post-1.0 |
| Payload accessors (typed) | ❌ Deferred | Typed payload register access. Raw union access is working. | Post-1.0 |

### Other API Features

| Feature | Status | Rationale | Target |
|---------|--------|-----------|--------|
| Multi-GPU | ❌ Deferred | Multiple CUDA devices. Infrastructure complexity (device selection, buffer mirroring) outweighs need for hobbyist renderer. | Post-1.0 |
| Demand-loaded textures | ❌ Deferred | OptiX sparse textures. Not needed — textures fit in VRAM for current use cases. | Post-1.0 |
| Cooperative vectors | ❌ Deferred | Neural rendering acceleration (OptiX 9.0 only). Bleeding-edge; no current use case. | Post-1.0 |
| Compilation cache | ⚠️ Internal | OptiX can cache compiled modules to disk. Not exposed as public API. | Post-1.0 |
| Logging callback | ⚠️ Internal | C++ logging callback configured internally. Not surfaced to Scala consumers. | Post-1.0 |
| Thread safety | ⚠️ Implicit | OptiX API is thread-safe per context. We use single-context model so it's naturally safe. | — |

---

## 1.0 Scope Boundary

**optix-jni 1.0 = stability contract on exposed API:**

| Category | In 1.0 | Post-1.0 |
|----------|--------|-----------|
| Acceleration structures | Triangles, spheres, cylinders, cones, planes, curves (cubic B-spline), recursive IAS | Motion blur, OMM, DMM, clusters |
| Curve variants | Cubic B-spline | Linear, quadratic, Catmull-Rom, ribbons |
| Denoising | HDR color-only + guides | Temporal, AOV, upscaling |
| SER | If merged in Sprint 30 | — |
| Validation mode | If merged in Sprint 30 | — |
| Shader programs | RG, CH, IS, MS | AH, EX, DC, CC |
| Instancing | 4×3 transforms, basic IAS | Visibility masks, SBT offsets, multi-level transforms |
| Multi-GPU | No | Maybe |
| Cooperative vectors | No | Maybe |

**Pre-1.0 checklist (from Task 30.5):**
- [ ] Public API review: keep/rename/deprecate for all public traits/classes/methods
- [ ] Scaladoc completeness CI gate
- [ ] MiMa baseline against 0.1.2 release
- [ ] Remove deprecated API
- [ ] 1.0 release checklist in optix-jni repo

---

*Generated as part of Sprint 30, Task 30.3. This document defines the scope of optix-jni 1.0 and should be placed in the optix-jni repo under `docs/api-audit.md` or as an arc42 §9 decision record.*
