# Sprint 14: Rendering Correctness & Code Health

**Sprint:** 14 - Rendering Correctness & Code Health
**Status:** In Progress
**Estimate:** 16–25 hours
**Branch:** `feature/sprint-14`
**Dependencies:** Sprint 12 (t-Parameter Animation) - required, Sprint 13 (Visual Quality) - optional

---

## Goal

Improve rendering correctness with caustics and multi-object colored shadow accumulation,
while addressing the most pressing code health issues identified during Sprint 13.

## Success Criteria

- [x] Code quality: OptiXEngine split below 400-line guideline
- [ ] Caustics render correctly (progressive photon mapping or alternative algorithm)
- [ ] Colored shadows Phase 2: multi-object anyhit accumulation works
- [ ] Documentation for caustics + colored shadows added to USER_GUIDE.md
- [ ] All tests pass

---

## Tasks

### Task 14.0: Code Quality — OptiXEngine Refactor ✓ COMPLETE

**Estimate:** 4–6 hours | **Actual:** ~1 hour

Split `createMultiObjectScene` (62 lines) and `rebuildScene` (61 lines) into 5 named private
sub-methods of ≤20 lines each: `configureRendererEnvironment`, `buildMixedSceneObjects`,
`buildInitialGeometry`, `applyRenderConfigAndFinalize`, `rebuildGeometry`. Everything stays
in `OptiXEngine.scala` (no new file). File reduced from ~430 → 391 lines. All 1683 tests pass.

#### Files Modified

- `menger-app/src/main/scala/menger/engines/OptiXEngine.scala` (430 → 391 lines)
- `CODE_IMPROVEMENTS.md` (M14 removed)

---

### Task 14.8: Colored Transparent Shadows Phase 2 (Multi-Object)

**Estimate:** 4–8 hours
**Dependency:** TD-6 (see `docs/arc42/11-risks-and-technical-debt.md`)

Enable colored shadow accumulation through multiple overlapping transparent objects. Phase 1
(Sprint 13.2) handles only the closest transparent object; this task adds anyhit-based
accumulation so that e.g. a red sphere behind a blue sphere each contribute their tint to
the shadow.

#### Background

Phase 1 uses a closesthit-only approach: the shadow ray stops at the first hit and returns
that object's RGB attenuation. Phase 2 requires an anyhit shader that multiplies running
attenuation across all transparent objects in front of the surface being lit, then terminates
only when the product is opaque or the ray exits the scene.

A previous anyhit attempt (reverted, see AD-8) failed due to:
- `optixTerminateRay`/`optixIgnoreIntersection` edge-case behavior at grazing angles
- Brightness changes when switching `calculateLighting()` from scalar to float3

These must be addressed carefully in isolation (do not mix with other shader changes).

#### Implementation Notes

- Add `__anyhit__shadow()` programs to sphere, triangle, and cylinder hit groups
- Accumulate per-channel attenuation multiplicatively: `atten *= alpha * (1 - color_rgb)`
- Terminate early when all channels are effectively opaque (threshold ≈ 0.99)
- Gate behind `transparent_shadows_enabled` flag — anyhit programs are no-ops when false
- Run full ShadowSuite + RendererTest before merging (the regression-prone suites from AD-8)

#### Tests to Add

- Two overlapping transparent spheres (different colors) produce combined tint in shadow
- Order independence: same two spheres reversed produce same shadow tint
- Opaque object behind transparent sphere still casts full shadow

---

### Task 14.9: Caustics

**Estimate:** 4–8 hours
**Dependency:** TD-4 (see `docs/arc42/11-risks-and-technical-debt.md`)

Revisit the caustics implementation from the preserved branch (deferred in Sprint 4).
Investigate and resolve the algorithm issues that caused incorrect results. If progressive
photon mapping (PPM) cannot be made correct within budget, document the findings and
consider an alternative approach (e.g. path-traced caustics, bidirectional path tracing stub).

#### Background

The original caustics branch used Progressive Photon Mapping. It was preserved but not merged
due to incorrect rendering results. The root cause has not been fully diagnosed.

#### Approach

1. Check out the caustics branch and identify the rendering artifacts
2. Diagnose: photon emission, gathering radius, or accumulation step?
3. Fix if feasible within 8h; otherwise document root cause in TD-4 and defer

#### Files (existing, from caustics branch)

- `optix-jni/src/main/native/shaders/caustics_ppm.cu`
- `docs/caustics/` — design notes from Sprint 4

---

### Task 14.10: Documentation

**Estimate:** 1 hour

Update USER_GUIDE.md and CHANGELOG.md for caustics and colored shadows Phase 2.

#### Sections to Add/Update

- Colored transparent shadows: multi-object accumulation behavior
- Caustics: usage and known limitations

---

## Summary

| Task | Description | Estimate | Priority |
|------|-------------|----------|----------|
| 14.0 | OptiXEngine refactor (code quality) | 4–6h | High |
| 14.8 | Colored shadows Phase 2 (multi-object anyhit) | 4–8h | Medium |
| 14.9 | Caustics | 4–8h | Medium |
| 14.10 | Documentation | 1h | High |
| **Total** | | **~13–23h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] USER_GUIDE.md updated
- [ ] CODE_IMPROVEMENTS.md: M14 removed (if 14.0 complete)

---

## Notes

### Deferred from This Sprint

The following tasks were originally planned for Sprint 14 but moved to later sprints:

- 14.1 Video output via ffmpeg → **Sprint 17**
- 14.2 Animation preview / t-scrubbing → **Sprint 17**
- 14.3 Soft shadows / area lights → **Sprint 15**
- 14.4 Depth of field / aperture → **Sprint 15**
- 14.5 New primitives: cylinder, cone, torus → **Sprint 15**
- 14.6 Coordinate cross / axis visualization → **Sprint 18**
- 14.7 Documentation → superseded by 14.10 (scoped to caustics + colored shadows only)
