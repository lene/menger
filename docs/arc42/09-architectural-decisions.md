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

## 9.2 Sprint-Level Decisions

Detailed implementation decisions are documented in sprint plans:

| Sprint | Plan | Key Decisions |
|--------|------|---------------|
| 5 | [SPRINT_5_PLAN.md](../../optix-jni/SPRINT_5_PLAN.md) | Triangle GAS, vertex format, per-face normals |
| 6 | [SPRINT_6_PLAN.md](../../optix-jni/SPRINT_6_PLAN.md) | IAS architecture, instance materials, sponge export |
| 7 | [SPRINT_7_PLAN.md](../../optix-jni/SPRINT_7_PLAN.md) | UV coordinates, texture support, material presets |

## 9.3 Future Decisions

Decisions to be made in upcoming sprints:

| Sprint | Topic | Options |
|--------|-------|---------|
| 8-10 | 4D projection method | Perspective vs orthographic vs cross-section |
| 11 | Scene file format | YAML vs JSON vs custom DSL |
| 12-13 | Animation keyframe format | Linear vs Bezier interpolation |
