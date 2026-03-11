# 9. Architectural Decisions

## 9.1 Decision Log

This section documents significant architectural decisions. Detailed implementation decisions are in sprint plans.

### AD-1: Surface Subdivision over Volume Subdivision

**Status:** Accepted
**Date:** Project inception

**Context:** Menger sponge generation requires recursive subdivision.

**Decision:** Use surface subdivision (O(12^n)) instead of volume subdivision (O(20^n)).

**Rationale:**
- Surface only generates visible outer faces
- No internal geometry (invisible faces)
- 60% complexity reduction per level

**Consequences:**
- Cannot render cross-sections (only surface visible)
- Requires separate algorithm for each dimension (3D, 4D)

---

### AD-2: Dual Rendering Pipeline (LibGDX + OptiX)

**Status:** Accepted
**Date:** 2025

**Context:** Need both interactive preview and high-quality output.

**Decision:** Maintain two parallel rendering paths:
- LibGDX (OpenGL) for real-time preview
- OptiX (ray tracing) for quality output

**Rationale:**
- LibGDX provides cross-platform, real-time performance
- OptiX provides physically-based rendering (refraction, caustics)
- Users can switch modes based on need

**Consequences:**
- Geometry must be exportable to both formats
- Some duplication in rendering code
- Independent camera controllers per path

---

### AD-3: Two-Layer OptiX Architecture

**Status:** Accepted
**Date:** 2025-11

**Context:** OptiX API is complex with many low-level details.

**Decision:** Separate into OptiXContext (low-level) and OptiXWrapper (high-level).

**Rationale:**
- OptiXContext: Pure API wrapper, testable, reusable
- OptiXWrapper: Scene state, convenience methods
- Clean separation of concerns

**Consequences:**
- More files to maintain
- Clear responsibility boundaries
- Easier testing of OptiX primitives

---

### AD-4: Scene Data in Params (not SBT)

**Status:** Accepted
**Date:** 2025-11

**Context:** SBT (Shader Binding Table) updates are expensive.

**Decision:** Store dynamic scene data (sphere color, IOR) in launch Params, not HitGroupData.

**Rationale:**
- Params can be updated without SBT rebuild
- Only geometry data in HitGroupData
- Faster iteration for material changes

**Consequences:**
- Limited to single object initially (Sprint 6 adds multi-object)
- Shader reads from params instead of SBT

---

### AD-5: JNI with Handle-Based Resource Management

**Status:** Accepted
**Date:** 2025

**Context:** Scala needs to manage C++ object lifetimes.

**Decision:** JNI layer uses handle (long pointer) stored in Scala object.

**Rationale:**
- Single handle per OptiXRenderer instance
- Explicit initialize()/dispose() lifecycle
- No hidden global state

**Consequences:**
- Must call dispose() to avoid leaks
- Each Scala object owns one C++ wrapper

---

### AD-6: Functional Error Handling (No Exceptions)

**Status:** Accepted
**Date:** Project inception

**Context:** Exceptions make code harder to reason about.

**Decision:** Use `Try`, `Either`, `Option` instead of exceptions.

**Rationale:**
- Explicit error handling in types
- Composable with map/flatMap
- Enforced by Wartremover

**Consequences:**
- More verbose in some cases
- Consistent error propagation
- JNI boundary requires careful translation

---

### AD-7: libGDX Wrapper Layer (`menger.gdx`)

**Status:** Accepted
**Date:** 2026-02

**Context:** LibGDX requires mutable state (`var` fields, `null` initialization) for camera objects, input tracking, and lifecycle management. This conflicted with the project's functional Scala style (Wartremover enforces no `var`/`null` outside wrappers).

**Decision:** Introduce `menger.gdx` as the sole package permitted to hold `var` and `null` for libGDX concerns. All Scala code outside this package interacts with immutable values and `Option[T]` instead of nullable references.

**Wrapper classes:**
- `GdxRuntime` — `Gdx.app` exit, rendering requests
- `KeyPressTracker` — Shift/Ctrl/Alt modifier state as `Boolean` vals
- `DragTracker` — mouse drag delta as immutable `(Float, Float)` snapshots
- `OrbitCamera` — spherical orbit math wrapping mutable libGDX `Vector3`

**Rationale:**
- Single, auditable boundary for mutability; easy to find all `var`s
- Input handlers and engines stay pure; testable without a running libGDX context
- Consistent with project rule: `var` only at necessary system boundaries

**Consequences:**
- Small overhead of wrapper delegation (negligible at interactive frame rates)
- New libGDX features must first be wrapped before use in non-`menger.gdx` code

---

### AD-8: Revert Anyhit RGB Shadow System (Sprint 13)

**Status:** Accepted (revert)
**Date:** 2026-03

**Context:** Sprint 13 attempted to implement colored transparent shadows using an
anyhit-based RGB shadow system. The implementation replaced the existing
closesthit-based scalar shadow pipeline with anyhit programs that computed per-channel
shadow attenuation (`float3` instead of `float`), enabling colored light transmission
through transparent objects (e.g., red glass casting red-tinted shadows).

**Decision:** Revert the anyhit RGB shadow system back to the original closesthit-based
scalar shadow pipeline.

**Rationale — why the implementation failed:**

1. **Scope coupling.** The shadow rewrite was mixed into the same working tree as an
   unrelated SBT offset fix for triangle mesh rendering. The two changes interacted in
   unexpected ways, making it difficult to isolate and debug each independently.

2. **Six regression failures.** The anyhit system caused failures in RendererTest (3 tests:
   semi-transparent blending, fully transparent rendering, combined optical effects) and
   ShadowSuite (3 tests: grazing angle, light-from-below, sphere-below-plane). Root causes
   included:
   - The anyhit shader's `optixTerminateRay()` / `optixIgnoreIntersection()` control flow
     changed shadow behavior for edge cases (grazing angles, degenerate geometries) that
     the closesthit pipeline handled correctly.
   - The `float3` shadow color required changes to `calculateLighting()` that altered
     brightness for semi-transparent and fully transparent spheres, breaking assertions
     on center brightness and background visibility.
   - Image size reductions (400x300 → 200x150) in `ThresholdConstants` compounded the
     brightness discrepancies by reducing pixel counts in validation regions.

3. **Payload encoding inconsistency exposed.** The SBT offset fix correctly routed shadow
   rays to geometry-specific hitgroups, exposing a latent bug where
   `__closesthit__triangle_shadow` used `optixSetPayload_0(1)` (raw integer) instead of
   `__float_as_uint(alpha)`. This was invisible before because triangle meshes in
   single-object mode accidentally used the sphere shadow shader.

**What was kept from the attempt:**
- `sbt_base_offset` field in Params and its use in all `optixTrace` calls (the SBT fix)
- `transparent_shadows_enabled` field in RenderConfig/Params (API surface, currently unused)
- `setTransparentShadows` JNI binding (dead code, available for future re-implementation)
- Fixed shadow payload encoding in `__closesthit__triangle_shadow` and
  `__closesthit__cylinder_shadow` (now use `__float_as_uint(alpha)`)

**Lessons for re-implementation:**
- Implement colored shadows as an isolated feature branch with no other changes mixed in.
- The anyhit approach is architecturally sound but requires careful handling of edge cases
  (grazing angles, degenerate geometry, fully transparent objects).
- Test against the full ShadowSuite and RendererTest before merging, not just new tests.
- Do not change test image sizes in the same change as shader behavior modifications.

**Consequences:**
- Shadows remain scalar (achromatic) — transparent objects cast gray shadows proportional
  to alpha, not colored shadows.
- The `setTransparentShadows(true)` API is a no-op until re-implemented.
- Triangle and cylinder shadow shaders now correctly encode alpha, fixing a latent bug.

---

## 9.2 Sprint-Level Decisions

Detailed implementation decisions are documented in sprint planning documents and code review files.
Historical sprint plans (5, 6, 7) have been archived after completion.

## 9.3 Future Decisions

Decisions to be made in upcoming sprints:

| Sprint | Topic | Options |
|--------|-------|---------|
| 12-13 | Animation keyframe format | Linear vs Bezier interpolation |

### Resolved Future Decisions

| Sprint | Topic | Decision |
|--------|-------|----------|
| 8-10 | 4D projection method | Perspective projection (implemented in `RotatedProjection`) |
| 11 | Scene file format | Scala DSL (implemented in Sprint 10; Sprint 11 used existing DSL) |
