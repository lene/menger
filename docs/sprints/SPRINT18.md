# Sprint 18: GPU Infrastructure

**Sprint:** 18 - GPU Infrastructure
**Status:** Not Started
**Estimate:** ~27 hours
**Branch:** `feature/sprint-18`
**Dependencies:** Sprint 17 (engine refactor, scene graph)

---

## Goal

Build the GPU infrastructure that enables advanced geometry and 4D rendering: multi-GAS IAS
for mixing geometry types, custom intersection (IS) program support, and GPU-side 4D math.
These three capabilities are tightly related (all involve SBT construction and program group
management) and benefit from being implemented together.

Ship at least one user-visible feature per infrastructure piece (recursive IAS sponge as
the first consumer of multi-GAS IAS), close out the half-implemented `maxRayDepth` work
from Sprint 17, and add a failed-render diagnostic so silent rendering failures stop
producing useless reference images.

## Success Criteria

- [ ] Multiple geometry types coexist in a single scene (spheres + meshes + cylinders)
- [ ] Custom intersection programs can be registered and used
- [ ] GPU-side 4D rotation, projection, and coordinate transforms work
- [ ] TD-5 (cannot mix spheres with triangle meshes) resolved
- [ ] Recursive-IAS Menger sponge renders at level 6 with VRAM ~ level-1 baseline
- [ ] `--max-ray-depth` CLI flag exposed and demonstrably affects glass-stack output
- [ ] All-uniform-pixel renders fail loudly with a diagnostic instead of silently saving
- [ ] All tests pass

---

## Tasks

### Task 18.1: Multi-GAS Instance Acceleration Structure (TD-5) — RESOLVED

**Estimate:** 8h (actual: ~1h — native side already in place)

Investigation during this sprint revealed that the native OptiX layer already
supported per-instance GAS handles: `OptiXWrapper::setTriangleMesh` appends to a
`triangle_meshes` vector, `addTriangleMeshInstance` builds a fresh GAS for each
appended mesh, and `buildIAS` already routes each instance to its own GAS via
`inst.gas_handle` and `inst.mesh_index`. The TD-5 limitation was therefore
purely a Scala-side over-conservative validation guard.

#### What Changed

- `menger-app/src/main/scala/menger/engines/scene/TriangleMeshSceneBuilder.scala`
  — `isCompatible` simplified: distinct triangle-mesh types coexist; only
  4D-projected pairs still require matching projection params (since projection
  is a global render setting).
- `menger-app/src/main/scala/menger/engines/SceneClassifier.scala` —
  sphere + multiple triangle-mesh types now classifies as `SimpleMixed`
  instead of `ComplexMixed`. The per-spec `setTriangleMesh +
  addTriangleMeshInstance` loop already does the right thing.
- Existing tests asserting the old rejection behaviour flipped to assert
  acceptance; new positive coverage added.
- `docs/arc42/11-risks-and-technical-debt.md` — TD-5 marked resolved.
- `scripts/manual-test.sh` — outdated TD-5 caveats removed.

#### Resolves

- **TD-5:** Cannot mix spheres with triangle meshes — RESOLVED
- Enables: scenes mixing spheres + cubes + sponges + tesseracts in a single
  render

---

### Task 18.2: Intersection Program Infrastructure — RESOLVED (doc-only)

**Estimate:** 4h (actual: ~1h — infrastructure already in place)

Investigation during this sprint revealed that the OptiX IS-program
infrastructure required for Sprint 19's analytical primitives is **already
implemented** in the cylinder pipeline:

- IS-program compilation lives in the umbrella shader module
  (`optix_shaders.cu` includes `hit_cylinder.cu`); CMake compiles it into
  the main PTX. No separate-module support is needed for new primitives.
- `PipelineManager` already registers IS + closesthit program-group pairs
  per geometry type (sphere, cylinder, triangle, plus shadow/photon
  variants).
- Per-primitive parameter passing already supports both styles: SBT-data
  (sphere) and `params.X_data[mat.texture_index]` indirection (cylinder).
  The latter is the recommended pattern for many-instance dynamic
  primitives.
- `HitGroupData` extensibility is provided by the per-primitive struct
  pattern (`CylinderData`, soon `ConeData`/`TorusData`/etc.).

Unifying these into a generic `PrimitiveParameters` abstraction was
considered and **deferred**: with only two existing patterns (sphere,
cylinder) the abstraction has no second data point to validate against.
Sprint 19's first new primitive will provide that data point and the
extraction can happen then.

#### What Changed

- `docs/dev/adding-analytical-primitives.md` (new) — recipe-style
  developer doc capturing the canonical params-indirection pattern
  (cylinder) as the reference for adding cone, torus, plane, parametric
  surfaces, etc. Includes step-by-step file-by-file recipe and a
  pre-PR checklist.

#### Out of scope

`PrimitiveParameters` unification — deferred until Sprint 19's second
analytical primitive lands and confirms the abstraction.

---

### Task 18.3: GPU 4D Transform and Projection ✅ DONE

**Estimate:** 5h
**Depends on:** 18.2

**Status:** Shipped on `feature/sprint-18` in six cuts:
- Cut A `0637af1` — native CUDA kernel + `setTriangleMesh4DQuads` setter (with
  ALLOW_UPDATE GAS flag and resident `d_quads_4d` for forward-compat).
- Cut B `d95f831` — `OptiXRenderer.setTriangleMesh4DQuads` Scala wrapper +
  `Mesh4DGpuFlatten` quad-buffer utility.
- Cut C `387101f` — `--gpu-project-4d` opt-in flag, `MeshUploadPlan`
  ADT, `MeshFactory.createUpload`, threaded through `TriangleMeshSceneBuilder`.
- Cut D `2930f6e` — `Project4DGpuSuite` static tests (CPU/GPU equivalence
  L∞ = 0/255 on tesseract no-rot + rotated; handcrafted non-tesseract
  Mesh4D generality; `tesseract-sponge level=2` setup ≈4s CPU vs ≈130ms
  GPU).
- Cut F `b463aaa` (API) + `84a91fe` (animation) — `updateMesh4DProjection`
  with in-place GAS+IAS refit; `WithAnimation` fast-path detects "only
  4D-projection params changed" frame-to-frame and re-projects via the
  update kernel instead of clearAllInstances+rebuild. 10-frame animation
  on tesseract-sponge level=2 = 5.5ms via update vs ≈1500ms via rebuild
  (~270×).
- Cut E (this commit) — integration/manual test entries + user-guide docs.

Move 4D rotation, projection, and coordinate transforms to GPU-side CUDA code.

**Current state:** 4D transforms computed on CPU (`Mesh4D`, `RotatedProjection`), projected
to 3D, sent as triangle geometry to OptiX.

**Target state:** 4D transforms evaluated per-ray on the GPU. This is the prerequisite
for procedural 4D geometry that doesn't need CPU-side tessellation.

#### What Changes

- CUDA device functions for 4D rotation matrices
- CUDA device functions for 4D-to-3D projection
- Launch parameters for 4D rotation angles and projection settings
- Test scenes validating GPU 4D output matches CPU reference

#### Migration Note

This begins the migration from CPU to GPU 4D math. The CPU path (`Mesh4D`,
`RotatedProjection`) becomes legacy and will be removed once all 4D geometry
is ported to the GPU path.

---

### Task 18.4: Recursive IAS Menger Sponge ✅ DONE

**Estimate:** 4h (actual: ~4h)
**Depends on:** 18.1
**Commits:** 71c892a (Cut A1 native scaffold), 7802530 (Cut A2 native bodies + JNI + Scala wrapper + smoke tests), 407c80a (Cut B CLI/scene-builder wiring), this commit (Cut C integration test + docs).

A 3D Menger sponge built as nested IAS rather than tessellated triangles: one
Level-1 GAS containing the 20 sub-cubes, then a Level-N IAS that references the
Level-(N−1) IAS 20 times with scale-1/3 + translate transforms. VRAM drops from
O(20ᴺ) triangles to O(N · 20) matrices, enabling level 10+ renders that are
otherwise impossible.

This is the first end-user feature consuming the 18.1 multi-GAS infrastructure;
without it, 18.1 ships as pure plumbing.

#### What Changes

- `menger-app/src/main/scala/menger/objects/SpongeRecursiveIAS.scala` (new) —
  scene description: emits the 20 generator transforms, requests N levels.
- `menger-app/src/main/scala/menger/optix/SceneConfigurator.scala` — recognise
  the `sponge-ias` object type and call the multi-level IAS builder.
- `optix-jni/src/main/native/OptiXContext.cpp` — IAS-of-IAS support if not
  already provided by 18.1.
- CLI: `--objects type=sponge-ias:level=N` parameter handling.
- Reference image + integration-test entry.

#### Out of scope

4D variant of recursive IAS — that is Sprint 21's job.

---

### Task 18.5: `maxRayDepth` CLI + verification

**Estimate:** 2h

The runtime path is **already wired** end-to-end (Scala `RenderConfig` →
`OptiXRenderer.setMaxRayDepth` → JNI → C++ `OptiXWrapper` → `params.max_ray_depth`
in shaders, all shipped before this sprint). What is missing:

- No `--max-ray-depth` CLI flag.
- Pipeline ceiling `MAX_TRACE_DEPTH = 5` is too low to test realistic glass
  stacks.
- No regression test proving the parameter actually changes rendered output.

#### What Changes

- `menger-app/src/main/scala/menger/MengerCLIOptions.scala` — add
  `--max-ray-depth` Scallop option, default `None`, range 1..ceiling.
- Wire CLI through `RenderSettings.maxRayDepth` to `RenderConfig`.
- `optix-jni/src/main/native/include/OptiXData.h` and `RenderConfig.scala` —
  raise `MAX_TRACE_DEPTH` / `RenderLimits.MaxRayDepth` (target 8) if the OptiX
  pipeline absorbs it without a stack-size change.
- `optix-jni/src/test/scala/menger/optix/MaxRayDepthSuite.scala` (new) — render
  a glass-stack scene at depths 2/4/8 and assert pixel-difference > epsilon
  between each pair.
- Integration-test entry exercising `--max-ray-depth` end-to-end.

#### Success Criterion

`--max-ray-depth N` produces visibly different renders across N for a
glass-stack scene; existing reference images unchanged at the default.

---

### Task 18.6: Failed-render diagnostic

**Estimate:** 1h

Detect frames that are uniformly one colour (typical failure modes: all-red
error fill, all-black no-trace, all-blue clear-colour) at save time, log a
diagnostic that includes the offending command, and exit non-zero so CI
catches the failure instead of saving a useless reference image.

#### What Changes

- `optix-jni/src/main/scala/menger/optix/RenderHealth.scala` (new) — pure
  helper: `check(pixels, width, height): Either[String, Unit]` flags a frame
  when ≥ 99 % of pixels are within ε of a single RGB value.
- Call site at PNG save: log `Failed render: all pixels are approximately
  (R,G,B); CLI args: ...`, delete the partial PNG, exit with status 2.
- Bypass: `--allow-uniform-render` CLI flag for legitimate uniform scenes
  (clear-colour smoke tests).
- Unit test: synthetic uniform pixel buffer triggers; varied buffer does not.

#### Success Criterion

A deliberately broken scene (no geometry, no background) produces a clear
error message and non-zero exit; existing integration tests unaffected.

---

### Task 18.7: Documentation

**Estimate:** 3h

- Sprint retrospective
- CHANGELOG.md update
- arc42 updates: ADs for multi-GAS IAS, IS program architecture, GPU 4D
  strategy, recursive instancing
- Developer docs for IS program registration API
- GPU 4D math API doc for later sprints
- User-guide entry for `--max-ray-depth`
- Failed-render-detector note in `debugging-rendering-bugs` skill (render-health section)

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 18.1 | Multi-GAS IAS (TD-5) | 8h | None |
| 18.2 | Intersection program infrastructure (doc-only — RESOLVED) | 1h | 18.1 |
| 18.3 | GPU 4D transform and projection | 5h | 18.2 |
| 18.4 | Recursive IAS Menger sponge | 4h | 18.1 |
| 18.5 | `maxRayDepth` CLI + verification | 2h | None |
| 18.6 | Failed-render diagnostic | 1h | None |
| 18.7 | Documentation | 3h | All |
| **Total** | | **~27h** | |

---

## Definition of Done

- [x] All success criteria met
- [x] All tests passing
- [x] Code quality checks pass: `sbt "scalafix --check"`
- [x] CHANGELOG.md updated (Unreleased — Sprint 18 entry covering 18.1, 18.3, 18.4, 18.5, 18.6)
- [x] arc42 updated with new architectural decisions (AD-17 multi-GAS IAS, AD-18 IS infra doc-only, AD-19 GPU 4D, AD-20 recursive IAS sponge)
- [x] IS program API documented for Sprint 19 implementors (AD-18; `OptiXContext::addProgramGroup` flagged as the public extension point)
- [x] GPU 4D math API documented for future sprint implementors (AD-19; user-guide §"GPU 4D Projection")
- [x] `--max-ray-depth` documented in user guide (§ Recursion Depth)
- [x] Failed-render diagnostic in `debugging-rendering-bugs` skill (render-health section)

---

## Retrospective

**What went well**
- The two-stage 18.3 delivery (one-shot upload first, per-frame update second)
  paid off: the forward-compat hooks baked in Cut A (resident `d_quads_4d`,
  `ALLOW_UPDATE` GAS, per-mesh `Projection4DParams`) survived contact with the
  actual animation refit path in Cut F without rework. ~270× animation speed-up
  on `tesseract-sponge level=2` (5.5 ms vs ≈1500 ms for 10 frames) validated
  the choice to land both stages in the same sprint.
- Multi-GAS IAS (18.1) shipped as quiet plumbing and immediately unlocked two
  user-visible features (recursive IAS sponge, GPU 4D refit). Sequencing
  infra-first was the right call.
- The render-health diagnostic (18.6) caught at least one bad reference image
  that would otherwise have entered the regression suite — it earned its keep
  in the same sprint it landed.
- Per-cut commits with the pre-commit hook running the full Scala suite kept
  every commit shippable; no rollbacks needed.

**What could go better**
- Sprint scope stretched to seven tasks (vs the original three in the
  "Why These Three Together" section). The expansion was justified ad-hoc;
  next time, surface the scope change explicitly when it happens rather than
  retrofitting the rationale.
- 18.2 ended as doc-only after audit. Worth flagging earlier: the audit could
  have been a one-day spike before the sprint planning lock-in.
- Manual-test entry numbering remains a live hazard (lesson burned in 18.4).
  We mitigated by appending; a stable-key scheme is worth a small dedicated
  task in a future sprint.

**Lessons carried forward**
- For continuous-update GPU APIs, ship the upload-time and update-time paths
  in the same sprint — the upload-time API design only firms up under
  pressure from the update-time consumer.
- Doc-only sprints (18.2, 18.7) earn their keep when they precede or finalise
  a code-heavy sprint; standalone doc sprints risk drift.

---

## Notes

### Why These Three Together

Multi-GAS IAS, IS programs, and GPU 4D math all involve the same infrastructure concern:
managing heterogeneous program groups in the SBT. Building multi-GAS IAS opens the SBT
construction logic; extending it for custom IS programs is a natural continuation;
GPU 4D math is the first consumer of custom IS programs for procedural geometry.

### Synergy with Sprint 19

Sprint 19 (Advanced Geometry) directly consumes all three capabilities:
- Multi-GAS IAS: coordinate cross (cylinders + other geometry), planes as geometry
- IS programs: analytical cylinder, cone, torus, plane intersection
- GPU 4D math: 4D polychora rendered via GPU-side projection

### Formerly

Content originally in Sprint 18 (Advanced Geometry) has moved to Sprint 19.

### Formerly Backlog

- **maxRayDepth** — Promoted into this sprint as task 18.5.
- Content originally in Sprint 20 (GPU 4D Infrastructure) has moved here as 18.3.
