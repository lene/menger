# Sprint 29: OptiX API Coverage I — Denoiser & Curves

**Sprint:** 29 - OptiX API Coverage I
**Status:** Not Started
**Estimate:** ~30 hours
**Branch:** `feature/sprint-29`
**Dependencies:** Sprint 27 (the optix-jni release that adds `updateTexture` — combine
the new API surface from this sprint into the same release train if timing allows)
**Feature ID:** F13 Phase 1 (includes F1) in [FEATURE_DEPENDENCIES.md](FEATURE_DEPENDENCIES.md)

---

## Goal

First phase of full OptiX API coverage for the `optix-jni` library: the AI denoiser
(the single highest-leverage unblock in the dependency map — it gates the future
progressive preview and improves IBL, area lights, and depth of field) and the curves
primitive (built-in ray-traced swept curves, which later sprints use for L-system stems,
streamtubes, and Hopf fibration fibers).

Both features land in `optix-jni` as generic library API plus menger-app integration,
and ship in the next optix-jni minor release.

---

## Success Criteria

- [ ] `RenderSettings(denoise = ...)` in DSL denoises the final accumulated frame;
      visibly less noise on an IBL `samples=1` scene at equal render time
- [ ] Denoising is off by default; reference images unchanged when off
- [ ] `Curve(points, radius)` DSL type renders a smooth swept tube via the OptiX curves
      primitive (no triangle mesh, no cylinder chain)
- [ ] Both APIs exposed in `optix-jni` as documented public API (Scaladoc)
- [ ] New optix-jni version published with denoiser + curves + `updateTexture`
- [ ] All tests pass

---

## Tasks

### Task 29.1: optix-jni — Denoiser API

**Estimate:** 8h

Wrap `OptixDenoiser` in the generic library.

**Implementation:**
- C++: create/destroy `OptixDenoiser` (model kind `OPTIX_DENOISER_MODEL_KIND_HDR`),
  scratch/state buffer management, `optixDenoiserInvoke` on the render buffer
- Guide layers: albedo + normal AOVs. Raygen writes first-hit albedo and shading
  normal into two additional output buffers (only when denoising is enabled — zero
  cost otherwise). Guide layers measurably improve edge preservation; plain-color
  denoising is the fallback when AOVs are unavailable
- Denoise in linear HDR **before** tone mapping — the denoiser model expects linear
  radiance; document this ordering constraint in the API
- JNI + Scala: `NativeOptiXApi.createDenoiser()`, `denoise(color, albedo, normal)`,
  `disposeDenoiser()`; Scala wrapper owns lifecycle via the existing AutoCloseable
  pattern
- Memory: denoiser state ~100–400 MB depending on resolution; allocate lazily on
  first use, document in optix-jni README

---

### Task 29.2: menger-app — Denoiser Integration

**Estimate:** 5h

**Implementation:**
- DSL: `RenderSettings(denoise: DenoiseMode = DenoiseMode.Off)` with
  `Off | Final` (`Final` = denoise once after all accumulation frames; per-frame
  modes are deferred to the progressive-preview feature)
- CLI: `--denoise` flag mapping to `DenoiseMode.Final`
- Pipeline order: accumulate (linear) → denoise → tone map → write PNG
- Render-health check interplay: a denoised uniform image is still uniform —
  no change needed, but add a test proving the failed-render diagnostic still fires
- Integration test: IBL scene `samples=1, accumulation=2` with and without
  `--denoise`; assert the denoised image differs and has lower pixel variance

---

### Task 29.3: optix-jni — Curves Primitive

**Estimate:** 8h

Wrap the OptiX built-in curve primitive.

**Implementation:**
- Build input `OPTIX_BUILD_INPUT_TYPE_CURVES`, primitive type
  `OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE` (round profile, smooth joints —
  the standard choice for tubes/strands)
- Vertex buffer (control points float3) + width buffer (per-vertex radius) +
  segment index buffer
- Pipeline: curve hit programs use the built-in intersector
  (`optixBuiltinISModuleGet`); add a curve hitgroup to the SBT alongside existing
  triangle/IS hitgroups
- Shading: curve normal from `optixGetCurveParameter` + position; reuse the existing
  material model (curves take the same `InstanceMaterial`)
- JNI + Scala: `addCurveInstance(points: Array[Float], widths: Array[Float],
  material): InstanceId` following the existing add*Instance pattern (use the opaque
  `InstanceId` from task 27.12)

---

### Task 29.4: menger-geometry / DSL — Curve Type

**Estimate:** 4h

**Implementation:**
- DSL: `Curve(points: Seq[Vec3], radius: Float | per-point radii, material)`
- Helper constructors: `Curve.helix(...)`, `Curve.circle(...)` for demo scenes
- Demo scene: trefoil knot tube (`examples.dsl.TrefoilKnot`) — exercises smooth
  closed-curve rendering
- Note for Sprint 35 (wireframe): once curves exist, wireframe edge rendering should
  evaluate curves vs. cylinder chains; leave a pointer in SPRINT35.md

---

### Task 29.5: Tests + Reference Images

**Estimate:** 3h

- Unit: denoiser lifecycle (mock-level), curve buffer validation (degenerate inputs:
  <4 control points, zero radius, NaN)
- Integration: denoise comparison scenario, trefoil knot reference image
- `scripts/manual-test.sh`: append denoise on/off pair and knot scene (append at end
  of file per repo policy)

---

### Task 29.6: Documentation + optix-jni Release

**Estimate:** 2h

- optix-jni README + Scaladoc for both APIs
- User guide: Denoising section (when to use, interplay with accumulation), Curves
  section
- Publish next optix-jni minor version (includes `updateTexture` from Sprint 27 work
  if not already released); bump the dependency in this repo
- CHANGELOG.md entry

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 29.1 | optix-jni denoiser API | 8h |
| 29.2 | menger-app denoiser integration | 5h |
| 29.3 | optix-jni curves primitive | 8h |
| 29.4 | DSL Curve type + demo | 4h |
| 29.5 | Tests + reference images | 3h |
| 29.6 | Docs + optix-jni release | 2h |
| **Total** | | **~30h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
- [ ] arc42 §5/§9 updated (denoiser pipeline stage, curves hitgroup)
- [ ] Integration + manual test scripts both cover denoise and curves
