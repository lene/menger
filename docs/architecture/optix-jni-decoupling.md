# optix-jni Decoupling: Architecture Decision

**Sprint:** 24–26
**Status:** Complete (Sprint 26)
**Decision Date:** 2026-05-29

---

## 1. Problem

`optix-jni` contains both generic GPU ray tracing infrastructure and Menger-specific
geometry (4D fractals, recursive sponge, caustics). It cannot be published as a
standalone JVM library because:

- Package `menger.optix` signals Menger ownership — generic users would not adopt it
- `Params` struct contains `Menger4DData`, `Sierpinski4DData`, etc. — Menger-specific
- `CausticsRenderer` is tuned to Menger sponge photon emission
- Depends on `menger-common` which also contains Menger-specific types

---

## 2. Target Architecture

```
┌─────────────────────────────────────────────────────────┐
│  io.github.lene:optix-jni                               │
│  Generic OptiX/CUDA ray tracing library                 │
│  - OptiX context, pipeline, SBT, optixLaunch            │
│  - Generic geometry: sphere, cylinder, cone, plane, mesh│
│  - BaseParams struct (no Menger-specific fields)        │
│  - Published: GitLab Package Registry + Maven Central   │
│  - Depends on: io.github.lene:menger-common_3           │
└───────────────────────┬─────────────────────────────────┘
                        │ dependsOn (published artifact)
┌───────────────────────▼─────────────────────────────────┐
│  menger-geometry  (in-repo, NOT published)              │
│  Menger-specific geometry layer                         │
│  - 4D fractals: Menger4D, Sierpinski4D, Hexadecachoron  │
│  - Recursive IAS sponge                                 │
│  - CausticsRenderer                                     │
│  - MengerParams struct (extends BaseParams)             │
└───────────────────────┬─────────────────────────────────┘
                        │ dependsOn
┌───────────────────────▼─────────────────────────────────┐
│  menger-app                                             │
│  Renderer, DSL, CLI, demos                              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  io.github.lene:menger-common_3                         │
│  Rendering primitives                                   │
│  - Color, Vector, Vec3, ImageSize, Light, Material      │
│  - Published: GitLab Package Registry + Maven Central   │
│  - No dependencies on other menger modules              │
└─────────────────────────────────────────────────────────┘
```

Both `optix-jni` and `menger-geometry` depend on `menger-common`. `menger-app` depends
on `menger-geometry` and `menger-common`.

---

## 3. Full OptiX API Scope

Goal: `io.github.lene:optix-jni` exposes OptiX 7.x primitives to JVM callers without
requiring Menger's scene graph or rendering pipeline.

**API style:** Thin JNI bindings (like LWJGL for Vulkan/OpenGL), not a higher-level
builder API. Each OptiX C function gets a corresponding JNI binding + Scala wrapper.

**Phase 1 (Sprint 25 — core launch path):**
- `OptixDeviceContext` — create, destroy, set log callback
- `OptixModule` — create from PTX/optixir source string
- `OptixProgramGroup` — create raygen, miss, hitgroup
- `OptixPipeline` — create, compile, link
- `OptixShaderBindingTable` — record setup utilities
- `optixLaunch` — synchronous and async variants
- `OptixAccelerationStructure` — build for triangles and AABB custom primitives

**Phase 2 (Sprint 26+):**
- `OptixDenoiser` — AI denoiser setup and invocation
- `OptixMotionTransform`
- `OptixCurves` (ribbon, round curves)
- Full traversal graph API

**Not in scope (ever):** Menger-specific geometry types, caustics, 4D projection.

**Target OptiX version:** 7.7+ (current in this project). API compatibility with 8.x
to be evaluated at upgrade time.

---

## 4. `BaseParams` / `MengerParams` Extension Pattern

### Problem

The OptiX launch parameters struct (`Params`) is a plain C struct passed by pointer
to `optixLaunch`. CUDA shaders receive it via `optixGetLaunchParamsPointer()` and
cast to the concrete type. The struct layout must be agreed upon at compile time — it
cannot be extended at runtime.

### Solution: C Struct Extension (Standard Pattern)

```cpp
// optix-jni: base parameters, generic geometry only
struct BaseParams {
    // ... all current generic fields (camera, lights, textures, etc.)
    // NO 4D fields, NO caustics
};

// menger-geometry: adds Menger-specific fields
struct MengerParams {
    BaseParams base;             // MUST be first — gives layout compatibility

    Menger4DData*         menger4d_data;
    unsigned int          num_menger4d;
    Sierpinski4DData*     sierpinski4d_data;
    unsigned int          num_sierpinski4d;
    Hexadecachoron4DData* hexadecachoron4d_data;
    unsigned int          num_hexadecachoron4d;
    CausticsParams*       caustics;
};
```

**Why this works:**
- `BaseParams` is at offset 0 in `MengerParams` (C standard guarantees this for
  the first struct member)
- Base shaders: `auto& p = *reinterpret_cast<BaseParams*>(optixGetLaunchParamsPointer())`
- 4D shaders: `auto& p = *reinterpret_cast<MengerParams*>(optixGetLaunchParamsPointer())`
- JNI launch in menger-geometry: `optixLaunch(..., sizeof(MengerParams), ...)`
- Both casts are valid as long as the launch allocated `sizeof(MengerParams)` bytes

**Validated prototype** (two-file C++ program, no CUDA required):
```cpp
// BaseParams.h
struct BaseParams { int a; float b; };

// MengerParams.h
struct MengerParams { BaseParams base; int c; };

// main.cpp
#include <cassert>
#include <cstddef>
int main() {
    MengerParams mp{}; mp.base.a = 42;
    BaseParams* bp = reinterpret_cast<BaseParams*>(&mp);
    assert(bp->a == 42);                    // field access through base pointer
    assert(offsetof(MengerParams, base) == 0);  // layout guarantee
}
```
This compiles and runs correctly under g++ and clang++.

---

## 5. Versioning Strategy

`optix-jni` and `menger-common` version **independently** from `menger-app`.

| Artifact | Version policy |
|----------|----------------|
| `menger-app` | Follows menger release (0.7.x, 0.8.x ...) |
| `menger-geometry` | Tracks menger-app (internal, same version) |
| `optix-jni` | Independent: starts at 0.1.0 on first external publication |
| `menger-common` | Independent: starts at 0.1.0 on first external publication |

Rationale: `optix-jni` and `menger-common` will have independent consumers after
publication. Their versions should not be tied to Menger's feature release cycle.

**SemVer for published artifacts:**
- MAJOR: breaking JVM API changes (class/method renames, removed types)
- MINOR: new functions/types, backward compatible
- PATCH: bug fixes, native-only changes with no JVM API change

---

## 6. Publication Pipeline

### Maven Central (via Sonatype Central Portal)

Plugin: `sbt-sonatype` 3.12+ (uses Central Portal API, not legacy OSSRH)

Requirements:
- GPG key: `043ADC824F884F3AE1CD5C8A11ED440B00A22863` (RSA 4096, created 2026-05-29)
- Namespace claim: `io.github.lene` — already verified on Central Portal
- Sonatype Central Portal account: registered at central.sonatype.com (username Yec7o3)
- Signed artifacts: `sbt-pgp` with key above; key published to keyserver.ubuntu.com

CI secrets required (GitLab CI variables):
- `PGP_PASSPHRASE` — GPG key passphrase
- `SONATYPE_USERNAME` — Central Portal token username
- `SONATYPE_PASSWORD` — Central Portal token password

### GitLab Package Registry

Uses standard Maven protocol. No extra plugin needed — same `publish` command with
GitLab Maven endpoint as `publishTo`.

CI: `CI_JOB_TOKEN` available automatically in all CI jobs.

Sprint 26 split the repositories in stages. Stage 1 published
`io.github.lene:menger-common_3:0.1.0` from the standalone `menger-common` repo to the
GitLab Package Registry and updated `menger` to resolve that artifact. Stage 2
published `io.github.lene:optix-jni:0.1.0` from the standalone `optix-jni` repo and
updated `menger` to consume both libraries from published artifacts.

### Native artifact in published JAR

The `.so` (shared library), `.ptx` (PTX shader), and native development API
(`optix-jni-native/include/**`, `optix-jni-native/shaders/**`) are bundled via
`resourceGenerators` in the standalone `optix-jni/build.sbt`. External consumers
load the native library via
`System.load(extractedPath)` using the JAR resource extraction pattern already
implemented in `OptiXRenderer.scala`.

Platform note: initial publication is Linux x86_64 only. Document this clearly.

---

## 7. Open Questions (resolved during Sprint 24)

| Question | Resolution |
|----------|-----------|
| GPG key available? | ✅ Yes — `043ADC824F884F3AE1CD5C8A11ED440B00A22863` (RSA 4096, 2026-05-29) |
| `io.github.lene` namespace on Maven Central? | ✅ Verified — already claimed on Central Portal |
| Full OptiX API: thin vs builder? | ✅ Thin JNI bindings (like LWJGL) |
| MengerParams layout safety? | ✅ Prototype validates struct extension pattern |
| Versioning: sync or independent? | ✅ Independent versioning starting at 1.0.0 |
