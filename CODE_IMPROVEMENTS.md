# Code Quality Improvements — Open Issues

**Last Updated:** 2026-05-31

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority

### H-jni-thrownew-null-class: FindClass not null-checked before ThrowNew in MengerJNIBindings.cpp

**Location**: `menger-geometry/src/main/native/MengerJNIBindings.cpp` — every catch block (7 sites)
**Impact**: High — `ThrowNew(nullptr, …)` is undefined behaviour; crashes in any environment where `java/lang/RuntimeException` is not on the class path (custom ClassLoader, stripped runtime).
**Effort**: 15 minutes (already fixed in sprint 25 via `if (exc)` guard on all sites)
**Status**: Fixed in sprint 25 — carried as note; pattern must not recur in new JNI files.

---

## Medium Priority

### M-texture-index-overloading: texture_index field has two incompatible semantics in InstanceMaterial

**Location**: `optix-jni/src/main/native/include/OptiXData.h:173–181`, `hit_cone.cu`, `hit_plane.cu`
**Impact**: Medium — semantic inconsistency; next developer adding a geometry type will copy the wrong precedent. **(judgment)**
**Effort**: 1–2 days (rename + audit all callsites)

`texture_index` in `InstanceMaterial` means different things for different geometry types:
- **Mesh/sphere**: image texture array index
- **Cone/plane**: geometry data array index (`cone_data[texture_index]` / `plane_data[texture_index]`)

The new `image_texture_index` field (added in Task 21.6) correctly separates image texture from geometry data for cone/plane — but `texture_index` itself remains semantically overloaded and the comment only partially explains the distinction:

```cpp
int texture_index;        // ← means image texture for mesh, geometry data for cone/plane
int image_texture_index;  // ← needed for cone/plane because texture_index is repurposed
```

Adding a third geometry type that needs both a geometry-data index AND an image texture index will require discovering this constraint by reading hit shaders.

**Direction**: Rename `texture_index` to `geometry_data_index` for cone/plane instances (it's only meaningful there), and use `image_texture_index` uniformly. Alternatively, split into typed fields: `cone_data_index`, `plane_data_index`, `image_texture_index`. Either path removes the semantic overload.

---

### M-arch-config-naming: *Config naming rule cannot be enabled — misplaced Config types

**Location**: `menger-app/src/test/scala/menger/ArchitectureSpec.scala` (ignored rule), multiple source files
**Impact**: Medium — misplaced types impede `optix-jni` extraction and future module splits.
**Effort**: 1–2 days

The ArchUnit rule enforcing `*Config` → `menger.config` or `menger.common` is ignored because these types are in the wrong packages:

| Type | Current | Correct |
|------|---------|---------|
| `InteractiveEngine.LevelConfig` | Inner class in `menger.engines` | Extract to top-level `menger.config` |
| `TAnimationConfig` | `menger.engines` | `menger.config` |
| `OrbitConfig` | `menger.input` | `menger.config` |
| `CausticsConfig`, `RenderConfig` | `menger.optix` | `menger.config` (optix imports from config, not vice versa) |
| `ProfilingConfig` | root `menger` | `menger.common` |

**Direction**: Migrate each type to its correct package (move + fix imports). `InteractiveEngine.LevelConfig` requires extracting it to a standalone file. After all are moved, remove the `ignore` from the rule in `ArchitectureSpec`.

---

### M-arch-archunit-case-class-field: ArchUnit haveOnlyFinalFields fires on Scala case class val fields

**Location**: `menger-app/src/test/scala/menger/ArchitecturePhase2Spec.scala` (ignored immutability rules)
**Impact**: Low — ArchUnit rule cannot be enabled even though there are no `var` fields; rule produces false positives.
**Effort**: 2–3 hours

Scala `val` fields in case classes compile to non-final JVM fields (the backing field is accessed via a getter, not declared `final`). ArchUnit's `haveOnlyFinalFields()` therefore flags every case class, making the immutability rule unusable as written.

Affected ignored rules:
- `menger.common` should have only final fields
- `menger.objects` should not use file IO or logging (Scala case classes implicitly implement `java.io.Serializable`, triggering the `java.io..` package check)
- `menger.common` should not use file IO (same Serializable issue)

**Direction**: Replace `haveOnlyFinalFields()` with a custom `DescribedPredicate[JavaField]` that checks for `var` fields by inspecting whether the field has a setter method in the same class. For the Serializable false-positives: add a `notSerializable` predicate to exclude `java.io.Serializable` specifically from the `java.io..` package check.

---

### M-arch-objects-logging: menger.objects uses SLF4J in geometry classes

**Location**: `menger-app/src/test/scala/menger/ArchitecturePhase2Spec.scala` (ignored rule), `menger/objects/higher_d/`
**Impact**: Low — logging in inner-layer geometry classes pulls infrastructure into the domain, complicating pure unit testing.
**Effort**: 2 hours

Two classes in `menger.objects` use SLF4J directly: `ParametricTessellator`, `higher_d/Rotation`. These are geometry-computation classes that should be pure (no I/O, no side effects). Logging is used for progress/debug output during tessellation. (`higher_d/Plane` and `higher_d/TesseractSponge2` had dead LazyLogging imports that have been removed.)

**Direction**: Remove `LazyLogging` from `ParametricTessellator` and `Rotation`. If progress reporting is needed, return metadata (e.g. triangle count, elapsed time) from the computation methods so callers (in `menger.engines`) can log it. Once removed, un-ignore the objects-logging rule in `ArchitecturePhase2Spec` (after also fixing the Serializable false-positive per M-arch-archunit-case-class-field).

---

### M-objectspec-optix-coupling: ObjectSpec (core domain) imports menger.optix.Material

**Location**: `menger-app/src/main/scala/menger/ObjectSpec.scala:9`
**Impact**: Medium — inverted module dependency; app's core domain type cannot be used without the JNI layer. **(judgment)**
**Effort**: 1–2 days (move Material to menger-common or define a mirror)

`ObjectSpec` is the primary app-side scene description type. It imports `menger.optix.Material`, placing a hard dependency from `menger-app`'s core domain into `menger-jni`. The intended direction is `menger-app → menger-jni`, but `ObjectSpec` specifically should be pure of JNI concerns — it's passed around in scene builders, test fixtures, and DSL converters, none of which need GPU types.

If a headless test or CLI module ever wants to validate or manipulate `ObjectSpec` without loading the native library, it can't.

**Direction**: Move `Material` (which is already a pure Scala enum/sealed trait) into `menger-common`. `menger-jni` can then import it from there. No native code changes needed.

---

### M-libgdx-wrapper-untested: LibGDX wrapper layer has no unit test coverage

**Location**: `menger-app/src/main/scala/io/github/lene/optix/` (`SceneConfigurator`, `OptiXRendererWrapper`, `CameraState`)
**Impact**: Medium — 0–17% coverage on the glue layer between OptiX and LibGDX; erratic lifecycle bugs are hard to reproduce and pin down. **(scheduled: Sprint 26, Task 26.8)**
**Effort**: 3h (headless LibGDX backend or interface extraction)

These classes are only instantiated inside `ApplicationListener.create()` / `render()`, so unit tests never touch them. Use LibGDX headless backend or extract testable logic behind interfaces. Target: ≥70% statement coverage on all three.

---

## Low Priority

| ID | Issue | Location |
|----|-------|----------|
| L-upload-texture-file-raw-int | `uploadTextureFromFile` returns raw `Int` (callers must check for negative); `uploadTexture` wraps in `Try`. Inconsistent error model at the same API boundary. | `optix-jni/src/main/scala/menger/optix/OptiXTextureApi.scala:45` |
| L-m4d-level-const | `addMenger4DInstance` hardcodes level bounds `0–14`; should be a named constant synced with Scala-side warnings | `OptiXWrapper.cpp:2313` |
| L-m4d-error-codes | `updateMenger4DProjection` returns -1/-2/-3 with no enum or comment mapping; future callers will guess | `OptiXWrapper.cpp`, `JNIBindings.cpp` |
| L-m4d-scene4d-sumtype | `Scene4DCache(gpu, cpu, menger4d)` are mutually exclusive but stored as three Options; a sealed trait would prevent multi-branch population at compile time | `InteractiveEngine.scala` |
| L-m4d-builder-validation | `Menger4DSceneBuilder` validates `level.isEmpty` but not out-of-range values; relies silently on `OptiXWrapper` to reject with -1 | `Menger4DSceneBuilder.scala` |
| L-jni-release-mode-readonly | `Release*ArrayElements` called with mode `0` (copy-back) for read-only arrays in `setCameraNative`, `setLights`, `setTriangleMeshNative`. Should use `JNI_ABORT`. No correctness bug, but wastes a copy. | `JNIBindings.cpp` (multiple sites) |
| L-cuda-synchronize-unchecked | `cudaDeviceSynchronize()` return value ignored at two call sites; errors silently discarded | `OptiXWrapper.cpp:628, ~718` |
| L-cuda-dispose-swallows | `dispose()` single try-catch swallows all exceptions, causing subsequent cleanup steps to be skipped if first step throws | `OptiXWrapper.cpp:3099` |
| L-tonemapping-magic-ints | ToneMapping operator encoded as magic int (0/1/2) split between `SceneConverter.scala:139–142` and `miss_plane.cu:92–97`; no compile-time sync between Scala enum and CUDA switch | `SceneConverter.scala`, `miss_plane.cu` |


## Tooling Gaps

*(none — all three gaps identified in the 2026-05-21 review have been closed)*

| Tool | Status | Where |
|------|--------|--------|
| ArchUnit | Closed — Phase 1: 14 active rules in `ArchitectureSpec.scala`; Phase 2: 5 active + 4 ignored-with-blockers in `ArchitecturePhase2Spec.scala`; wired into `sbt test` | `menger-app/src/test/scala/menger/ArchitectureSpec.scala`, `ArchitecturePhase2Spec.scala` |
| cppcheck | Closed — runs in pre-push hook + CI `Test:Cppcheck` job | `.cppcheck-suppress`, `.git_hooks/pre-push`, `.gitlab-ci.yml` |
| clang-tidy | Closed — `compile_commands.json` enabled via CMake; runs in pre-push hook + CI `Test:ClangTidy` job | `.clang-tidy`, `CMakeLists.txt`, `.git_hooks/pre-push`, `.gitlab-ci.yml` |

---

## Feature Ideas (Sprint 20+)

These are deferred feature ideas, not defects.

| ID | Idea | Location | Est. Hours |
|----|------|----------|------------|
| L2 | Metrics and Telemetry | New feature | 6-8 |
| L3 | Scene graph abstraction | Architecture | 10-12 |
| L4 | Comprehensive benchmarking suite | Tests | 8-10 |
| L5 | Plugin system for geometry types | Architecture | 12-15 |

---

## Accepted / Deferred

Issues that were investigated and consciously accepted:

| Item | Decision |
|------|----------|
| Mutable state in LibGDX integration | Required by LibGDX framework |
| M11: Input controller mutable state | Well-structured; encapsulation adds complexity without benefit |
| L11: Exceptions in CudaBuffer (CudaBuffer.h:77,89) | Correct pattern at JNI boundaries |
| OptiX cache management | Works correctly |
| Caustics algorithm limitations | Resolved in Sprint 14 (PPM implemented; remaining limits documented in `docs/guide/advanced.md` §Caustics) |
| L-film-blend: blendFresnelColorsRGBAndSetPayload duplicates scalar body | GPU perf trade-off; acceptable if documented |
| OptiX DSL runtime evaluation | Deferred (Sprint 15) |
| H-glass-sponge-skin-diffuse | Sprint 17: `use_coverage_blend` now excludes refractive materials; `use_refractive_coverage_blend` path added (vertex_alpha × Fresnel + (1−α) × continuation); `maxRayDepth` implemented in JNI/shader. Full investigation (glass-sponge-investigation.md) found remaining visible artifacts are physically correct Fresnel reflection of the pink background at grazing angles — not a code bug. Closed. |
| L-cli-monolith: MengerCLIOptions is a 375-line monolith | Scallop registers options during construction; extracting groups into separate `self: ScallopConf =>` traits risks initialization-order issues. File is already organized with clear group separators; accept as-is. |
| L-cli-validation-density: CliValidation repetitive requires-pattern | `isSupplied` must be evaluated lazily inside `validateOpt` lambdas (after argument parsing), not eagerly in a data-driven list. The repetition is load-bearing; accept as-is. The `case Some(_)/None` branches were simplified to `case _` where safe. |
| M-film-maxdepth-opaque-fallback: Film opaque at max_ray_depth | Unconfirmed. `use_refractive_coverage_blend` requires `has_vertex_alpha_channel`; plain Film geometry (spheres, parametric) has no vertex alpha and never enters that branch. No existing scene combines Film + vertex-alpha geometry to trigger the hypothesised fallback. |
