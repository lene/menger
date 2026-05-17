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

**Status:** Superseded by AD-16 (OptiX-only from v0.6)
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

**Update (Sprint 13.2):** Phase 1 of colored transparent shadows has been re-implemented
using a closesthit-only approach (no anyhit programs), resolving the first two consequences
above. Shadows are now RGB when `transparent_shadows_enabled = true`, and the
`setTransparentShadows` API is functional. The anyhit programs and unused overloads from
the failed attempt have been removed. See `08-crosscutting-concepts.md` §8.1.4 for the
colored shadow algorithm and TD-6 for remaining Phase 2 work.

---

### AD-16: OptiX as Sole Renderer (Remove LibGDX 3D Rendering)

**Status:** Accepted
**Date:** 2026-04 (Sprint 17)

**Context:** The dual rendering pipeline (AD-2) was originally adopted to maintain both a real-time
LibGDX/OpenGL preview and a high-quality OptiX path. As OptiX has matured to provide interactive
frame rates (>100 FPS for typical scenes), the LibGDX 3D rendering path became redundant. It added
maintenance burden, code complexity, and required a separate `--optix` flag to access the higher
quality renderer.

**Decision:** Remove the LibGDX 3D rendering path entirely. OptiX is now the sole renderer.
LibGDX is retained as the windowing/GL framework (LWJGL3 application lifecycle, input handling),
but the LibGDX 3D scene graph, `ModelFactory`, `MengerEngine`, `InteractiveMengerEngine`, and
`AnimatedMengerEngine` are deleted. The `--optix` CLI flag is removed; all rendering always uses
OptiX.

**Rationale:**
- OptiX achieves interactive frame rates making a fast-preview mode unnecessary
- Eliminates duplicated geometry export code and engine branching
- Simplifies the CLI (no `--optix` flag required)
- Reduces maintenance surface by removing ~2,000 lines of LibGDX 3D code
- Improves render quality for all users by default

**Consequences:**
- Users must provide `--objects` or `--scene` to render (no default object)
- All rendering is physically-based ray tracing; no fast OpenGL fallback
- LibGDX 3D scene graph and `ModelFactory` deleted; `menger.gdx` retained for windowing/input wrappers
- Supersedes AD-2

---

### AD-17: Multi-GAS Instance Acceleration Structure (Sprint 18.1)

**Status:** Accepted
**Date:** 2026-04 (Sprint 18.1)

**Context:** Up to Sprint 17 the renderer compiled every scene object into a
single Geometry Acceleration Structure (GAS). Heterogeneous scenes (mixed
triangle and procedural geometry, mixed materials, recursive sponges) had
to be flattened into one builder, which prevented per-object material
inheritance, per-object IS programs, and any form of geometry instancing.

**Decision:** Promote the top-level traversable from a single GAS to an
Instance Acceleration Structure (IAS). Each scene object owns a private GAS
that is referenced from the IAS by an `OptixInstance` carrying its
transform and SBT-offset. The IAS is rebuilt when the scene changes; GASes
are reused across frames.

**Rationale:**
- Decouples per-object data from the global SBT — each instance can carry
  its own material slot.
- Enables nested instancing (Sprint 18.4 recursive sponge — IAS-of-IAS).
- Required for any per-object procedural geometry (Sprint 18.2 IS programs)
  without exploding the SBT.
- Standard OptiX pattern; no novel mechanics.

**Consequences:**
- The hit shaders look up material data via `optixGetInstanceId()` instead
  of a single global pointer.
- A small constant per-frame cost (IAS build) added; amortised by GAS
  reuse and dwarfed by triangle counts.
- Opens the door for AD-18 (IS programs) and AD-20 (recursive IAS sponge),
  which would not be feasible on a flat single-GAS layout.

---

### AD-18: Intersection Program Infrastructure (Sprint 18.2 — doc-only)

**Status:** Accepted (infrastructure-only, no end-user feature)
**Date:** 2026-04 (Sprint 18.2)

**Context:** Procedural primitives in OptiX (spheres, cylinders, custom
SDFs) require user-supplied intersection (IS) programs registered as
program groups in the SBT. Sprint 18.2 audited what the existing pipeline
already supports.

**Decision:** No code change in 18.2. The Sprint 18.1 multi-GAS IAS
together with the existing program-group registration path already covers
heterogeneous IS programs per object. The audit ships as a developer-doc
ADR rather than a code commit; future procedural geometry sprints
(Sprint 19 n-gon Mesh4D, future SDF work) plug into the same
`OptiXContext::addProgramGroup` API used by the built-in sphere and
cylinder IS programs.

**Rationale:**
- Avoids speculative infrastructure that no end-user feature was ready to
  consume.
- Keeps the SBT layout decision visible (one program group per IS variant)
  for the implementor of the first new procedural primitive.

**Consequences:**
- Sprint 19 owns the first new IS program registration; no abstractions
  invented in advance.
- The `OptiXContext` SBT layout is documented as the public extension
  point.

---

### AD-19: GPU-Side 4D Projection Strategy (Sprint 18.3)

**Status:** Accepted
**Date:** 2026-04 (Sprint 18.3)

**Context:** Until Sprint 18.3 every 4D-to-3D projection (rotate XW/YW/ZW,
perspective divide, face-normal recompute) was performed on the CPU at
scene-build time. For deep tesseract sponges (level 3+) this dominated
setup time; per-frame animation of 4D rotation forced a full scene rebuild.

**Decision:** Add a plain `__global__` CUDA kernel
(`project4d_quads_kernel`) that runs **once per quad in parallel**,
reusing the resident `d_quads_4d` device buffer. The kernel produces the
same stride-8 vertex layout as the CPU path, then feeds the GAS builder
unchanged. Two API surfaces:

- `setTriangleMesh4DQuads` — one-shot upload + projection, produces a
  GAS with `OPTIX_BUILD_FLAG_ALLOW_UPDATE` set on top of the existing
  flags.
- `updateMesh4DProjection` — re-launch the kernel against the same
  buffers, then refit the GAS and parent IAS in-place
  (`OPTIX_BUILD_OPERATION_UPDATE`).

The flag `--gpu-project-4d` opts in. Default behaviour stays bit-for-bit
identical to the CPU path. Animation drivers detect "frame-to-frame only
the 4D-projection params changed" and call the update kernel instead of
rebuilding the scene.

**Rationale:**
- Quads (not arbitrary 4D triangles) preserve bit-for-bit equivalence
  with the existing CPU pipeline as the cheapest correctness oracle.
- One CUDA thread per quad → embarrassingly parallel, no atomics, no
  SBT involvement.
- Forward-compatible: variable-arity polygons (Sprint 19) plug in via a
  sibling kernel; the animation-refit path is shape-agnostic.

**Consequences:**
- First non-OptiX CUDA kernel in the codebase (`project4d.cu`); CMake
  picks it up automatically alongside the OptiX shader compile.
- 4D-projected meshes carry an extra `Projection4DParams` sub-struct in
  `triangle_meshes[i]`; CPU-uploaded meshes are unaffected
  (sentinel `num_quads == 0`).
- Measured speed-up on `tesseract-sponge level=2`: ~30× setup, ~270×
  animation (10-frame XW rotation). Equivalence verified L∞ = 0/255 on
  static frames, ≤ 6/255 over the animation envelope.

---

### AD-20: Recursive IAS for Menger Sponge (Sprint 18.4)

**Status:** Accepted
**Date:** 2026-04 (Sprint 18.4)

**Context:** A Menger sponge of recursion level N tessellates to O(20ᴺ)
triangles. Levels above 5 are unrenderable on any GPU (24+ GB of
triangles). The 18.1 multi-GAS IAS exposes nested instancing as an
alternative.

**Decision:** Add a `sponge-recursive-ias:level=N` object type that
builds the sponge as nested IAS layers: one Level-1 GAS containing the
20 sub-cubes, then a Level-N IAS that references the Level-(N−1) IAS
20 times with scale-1/3 + translate transforms. VRAM scales as
O(N · 20) matrices instead of O(20ᴺ) triangles.

**Rationale:**
- Enables levels 6..14 (capped by `OPTIX_MAX_TRAVERSABLE_GRAPH_DEPTH`)
  that the triangle path cannot represent at all.
- Reuses the multi-GAS IAS infrastructure shipped in 18.1 — no new
  builder, no new SBT layout.

**Consequences:**
- The first end-user feature consuming AD-17. Without 18.4, AD-17 would
  ship as pure plumbing.
- Lighting/shadow appearance differs subtly from the tessellated path
  because nested IAS instances do not share triangle-level normals; this
  is documented in the user-guide and accepted as part of the trade-off
  for unbounded depth.
- Recursive 4D variant (tesseract-sponge as IAS) is deferred to a later
  sprint.

---

### AD-21: OptiXRenderer Facade Split into Responsibility Traits (Sprint 20)

**Status:** Accepted
**Date:** 2026-05 (Sprint 20)

**Context:** `OptiXRenderer.scala` had grown to ~995 lines, mixing JNI bindings for
sphere management, mesh upload, plane configuration, texture handling, and render
control into a single class. This violated the single-responsibility principle and
made it hard to locate bindings for a specific subsystem.

**Decision:** Extract five `private[optix]` traits, each covering one responsibility
area, and reduce `OptiXRenderer` to a thin class that mixes them all in:
- `OptiXSphereApi` — sphere add/configure JNI bindings
- `OptiXMeshApi` — triangle mesh upload/configure
- `OptiXPlaneApi` — plane add/configure
- `OptiXTextureApi` — texture upload, procedural textures, PBR map wiring
- `OptiXRenderApi` — render, camera, lights, scene management

**Rationale:**
- Each trait is locatable in one file; navigation time drops sharply.
- `private[optix]` visibility keeps the split internal; external callers see
  `OptiXRenderer` unchanged.
- No JNI binding changes — purely a Scala-layer reorganization.

**Consequences:**
- `OptiXRenderer.scala` is now ~180 lines; each Api trait ~60–150 lines.
- New JNI bindings go in the matching trait, not the catch-all class.
- Section 2.5.1 constraint (`nativeHandle` var) still applies to `OptiXRenderer`.

---

### AD-22: ObjectSpec Value Object Extraction (Sprint 20)

**Status:** Accepted
**Date:** 2026-05 (Sprint 20)

**Context:** `ObjectSpec` accumulated flat fields for rotation angles, cone geometry,
plane geometry, procedural spec, and texture maps — 15+ fields in a single case class
with no grouping. Adding new fields (e.g., Sprint 20 PBR maps) required touching the
main constructor and every callsite.

**Decision:** Extract five typed value objects as nested members of `ObjectSpec`:
- `ObjectRotation(x, y, z: Float)` — 3D rotation angles in degrees
- `ConeGeometry(apex, base, radius)` — cone shape parameters
- `PlaneGeometry(normal, distance, color2, checkerSize)` — plane shape/material
- `ProceduralSpec(proceduralType: Int, scale: Float)` — procedural texture config
- `TextureMaps(normalMap, roughnessMap: Option[String])` — PBR map file references

`ObjectSpec` exposes convenience forwarders (`def rotX`, `def normalMap`, etc.) so
existing callsites compile unchanged.

**Rationale:**
- Grouped semantics reduce constructor length and cognitive load.
- New fields in a group require a single value-object change, not an `ObjectSpec` diff.
- Per-aspect private parse helpers (`parseRotation3DDeg`, `parseConeGeometry`, …)
  align with the value objects, making `parse()` a straightforward composition.

**Consequences:**
- `ObjectSpec.parse()` delegates to per-aspect helpers; error messages carry context.
- `SceneObject` DSL case classes expose the same fields (`normalMap`, `roughnessMap`,
  `proceduralType`, `proceduralScale`) via `baseObjectSpec()`, which wraps them into
  `ProceduralSpec` / `TextureMaps` at the boundary.

---

## 9.2 Sprint-Level Decisions

Detailed implementation decisions are documented in sprint planning documents and code review files.
Historical sprint plans (5, 6, 7) have been archived after completion.

## 9.3 Future Decisions

Decisions to be made in upcoming sprints:

No pending future decisions at this time.

### Resolved Future Decisions

| Sprint | Topic | Decision |
|--------|-------|----------|
| 8-10 | 4D projection method | Perspective projection (implemented in `RotatedProjection`) |
| 11 | Scene file format | Scala DSL (implemented in Sprint 10; Sprint 11 used existing DSL) |
| 12-13 | Animation keyframe format | Deferred indefinitely (animation not yet scheduled) |
| 15 | Area light shadow sampling | Per-light `shadowSamples` in `area:` CLI spec (not global `--shadow-samples` flag) |
