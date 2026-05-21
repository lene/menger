# Code Quality Improvements — Open Issues

**Last Updated:** 2026-05-21

Resolved items are removed from this file entirely — git history is the record of what was fixed.

---

## High Priority

*(none)*

---

## Medium Priority

### M-jni-buffer-pin-leak: JNI array pinning leak in renderWithStats exception path

**Location**: `optix-jni/src/main/native/JNIBindings.cpp:832–863`
**Impact**: Medium — memory-safety issue on the render error path; repeated render failures leak JVM heap.
**Effort**: 1 hour

`GetByteArrayElements` pins the render output array so the GPU can write into it. If `wrapper->render()` throws a C++ exception, control jumps to the `catch` block at line 863 and returns `nullptr` — without calling `ReleaseByteArrayElements`. The JVM's pinned array is never unpinned and never GC'd.

```cpp
jbyte* buffer = env->GetByteArrayElements(imageArray, nullptr);
// ...
wrapper->render(width, height, reinterpret_cast<unsigned char*>(buffer), &stats);
// ← if this throws, ReleaseByteArrayElements is never called
env->ReleaseByteArrayElements(imageArray, buffer, 0);
```

Render failures are uncommon in steady-state operation but do happen (OptiX pipeline rebuild, VRAM exhaustion), and the leak is proportional to image size (width × height × 4 bytes per leak). In long-running interactive sessions this will silently exhaust JVM heap.

**Direction**: Wrap the `GetByteArrayElements`/`render`/`Release` block in RAII or use `GetPrimitiveArrayCritical` with a scoped release. Simplest fix: call `ReleaseByteArrayElements(imageArray, buffer, JNI_ABORT)` in the catch block before returning.

---

### M-jni-init-swallowed-exception: initializeNative swallows C++ exception, no Java exception thrown

**Location**: `optix-jni/src/main/native/JNIBindings.cpp:71–76`
**Impact**: Medium — diagnosis gap; on OptiX initialization failure, Scala sees `false` but no exception message.
**Effort**: 30 minutes

Every other JNI function in JNIBindings.cpp converts C++ exceptions to Java exceptions via `env->ThrowNew(...)`. `initializeNative` is the sole exception: its catch block logs to stderr and returns `JNI_FALSE` silently.

```cpp
} catch (const std::exception& e) {
    std::cerr << "[JNI] Error in initializeNative: " << e.what() << std::endl;
    return JNI_FALSE;  // ← no ThrowNew; Scala gets false with no exception
}
```

Callers that check `ensureAvailable()` get `OptiXNotAvailableException("Failed to initialize OptiX renderer")` with no root cause. The actual OptiX error (device not found, driver mismatch, etc.) is lost unless someone is watching stderr.

**Direction**: Call `env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what())` before returning, matching the pattern of every other function in this file.

---

### M-render-null-type-contract: renderWithStats declares RenderResult but may return null

**Location**: `optix-jni/src/main/scala/menger/optix/OptiXRenderApi.scala:33–38`
**Impact**: Medium — type contract violation; callers without defensive wrapping NPE silently.
**Effort**: 2–3 hours

The Scala return type is `RenderResult` (non-nullable by convention), but the method returns `null` on JNI failure:

```scala
def renderWithStats(size: ImageSize): RenderResult =
  // ...
  Option(raw).map(_.copy(frameMs = elapsedMs)).orNull  // ← null leaks into typed API
```

Two callers handle this defensively (`Option(renderWithStats(size)).map(...)`) but any future caller using the result directly will NPE. The type signature gives no indication that null is possible.

**Direction**: Change return type to `Option[RenderResult]` and update all callers. The two existing defensive callers already do `Option(...)` so the change mechanically removes one layer. The third call site (in `InteractiveEngine`) needs inspection to confirm it handles None correctly.

---

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

### M-objectspec-optix-coupling: ObjectSpec (core domain) imports menger.optix.Material

**Location**: `menger-app/src/main/scala/menger/ObjectSpec.scala:9`
**Impact**: Medium — inverted module dependency; app's core domain type cannot be used without the JNI layer. **(judgment)**
**Effort**: 1–2 days (move Material to menger-common or define a mirror)

`ObjectSpec` is the primary app-side scene description type. It imports `menger.optix.Material`, placing a hard dependency from `menger-app`'s core domain into `menger-jni`. The intended direction is `menger-app → menger-jni`, but `ObjectSpec` specifically should be pure of JNI concerns — it's passed around in scene builders, test fixtures, and DSL converters, none of which need GPU types.

If a headless test or CLI module ever wants to validate or manipulate `ObjectSpec` without loading the native library, it can't.

**Direction**: Move `Material` (which is already a pure Scala enum/sealed trait) into `menger-common`. `menger-jni` can then import it from there. No native code changes needed.

---

## Low Priority

| ID | Issue | Location |
|----|-------|----------|
| L-upload-texture-file-raw-int | `uploadTextureFromFile` returns raw `Int` (callers must check for negative); `uploadTexture` wraps in `Try`. Inconsistent error model at the same API boundary. | `optix-jni/src/main/scala/menger/optix/OptiXTextureApi.scala:45` |
| L-m4d-level-const | `addMenger4DInstance` hardcodes level bounds `0–14`; should be a named constant synced with Scala-side warnings | `OptiXWrapper.cpp:2313` |
| L-m4d-error-codes | `updateMenger4DProjection` returns -1/-2/-3 with no enum or comment mapping; future callers will guess | `OptiXWrapper.cpp`, `JNIBindings.cpp` |
| L-m4d-scene4d-sumtype | `Scene4DCache(gpu, cpu, menger4d)` are mutually exclusive but stored as three Options; a sealed trait would prevent multi-branch population at compile time | `InteractiveEngine.scala` |
| L-m4d-builder-validation | `Menger4DSceneBuilder` validates `level.isEmpty` but not out-of-range values; relies silently on `OptiXWrapper` to reject with -1 | `Menger4DSceneBuilder.scala` |


## Tooling Gaps

- **No ArchUnit** (or equivalent): Module boundary rules (e.g., "menger-jni must not import menger-app") are enforced by review convention, not by a failing test. Finding M-objectspec-optix-coupling would have been caught automatically. ArchUnit operates on JVM bytecode and works with Scala; rules can be expressed as JUnit tests wired into `sbt test`.
- **cppcheck not installed**: Native-side static analysis has no automated layer beyond compiler warnings. Consider `apt install cppcheck` or adding it to the project's dev-environment setup.
- **No compile_commands.json**: clang-tidy cannot run. CMake can generate one with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`.

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
