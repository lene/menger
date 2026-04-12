# Sprint 18: GPU Infrastructure

**Sprint:** 18 - GPU Infrastructure
**Status:** Not Started
**Estimate:** ~19 hours
**Branch:** `feature/sprint-18`
**Dependencies:** Sprint 17 (engine refactor, scene graph)

---

## Goal

Build the GPU infrastructure that enables advanced geometry and 4D rendering: multi-GAS IAS
for mixing geometry types, custom intersection (IS) program support, and GPU-side 4D math.
These three capabilities are tightly related (all involve SBT construction and program group
management) and benefit from being implemented together.

## Success Criteria

- [ ] Multiple geometry types coexist in a single scene (spheres + meshes + cylinders)
- [ ] Custom intersection programs can be registered and used
- [ ] GPU-side 4D rotation, projection, and coordinate transforms work
- [ ] TD-5 (cannot mix spheres with triangle meshes) resolved
- [ ] All tests pass

---

## Tasks

### Task 18.1: Multi-GAS Instance Acceleration Structure (TD-5)

**Estimate:** 8h

Build an IAS (Instance Acceleration Structure) containing multiple GAS instances,
each with potentially different geometry types and program groups.

#### What Changes

- `OptiXContext`: support building multiple GAS (one per geometry type)
- `OptiXContext`: build IAS from GAS instances with transforms
- `OptiXWrapper`: scene state management for heterogeneous geometry
- SBT construction: multiple hitgroup entries (one per geometry type per ray type)
- `SceneConfigurator`: build multi-GAS scenes from scene graph
- `JNIBindings`: expose multi-geometry scene setup

#### Resolves

- **TD-5:** Cannot mix spheres with triangle meshes
- Enables: planes as first-class geometry, coordinate cross, mixed scenes

---

### Task 18.2: Intersection Program Infrastructure

**Estimate:** 4h
**Depends on:** 18.1

Add support for custom OptiX intersection programs (IS programs). This is the shared
infrastructure used by both 3D analytical primitives (Sprint 19) and 4D parametric
surfaces (Sprint 20+).

#### What Changes

- `OptiXContext`: IS program compilation and registration
- SBT: support IS + closesthit program pairs per geometry type
- `HitGroupData`: extensible per-primitive parameter passing
- Define `PrimitiveParameters` structure for analytical shapes (center, radius, axis, etc.)

#### Design Constraint

This infrastructure must be general enough for:
- 3D analytical primitives: cylinder, cone, torus, plane (Sprint 19)
- 4D parametric surfaces: `f(u,v) -> Vec4` projected per-ray (future)
- Volume intersection: ray marching for scalar fields (future)

---

### Task 18.3: GPU 4D Transform and Projection

**Estimate:** 5h
**Depends on:** 18.2

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

### Task 18.4: Documentation

**Estimate:** 2h

- Sprint retrospective
- CHANGELOG.md update
- arc42 updates: AD for multi-GAS IAS, IS program architecture, GPU 4D strategy
- Developer docs for IS program registration API
- Document GPU 4D math API for use in later sprints

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 18.1 | Multi-GAS IAS (TD-5) | 8h | None |
| 18.2 | Intersection program infrastructure | 4h | 18.1 |
| 18.3 | GPU 4D transform and projection | 5h | 18.2 |
| 18.4 | Documentation | 2h | All |
| **Total** | | **~19h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] arc42 updated with new architectural decisions
- [ ] IS program API documented for Sprint 19 implementors
- [ ] GPU 4D math API documented for future sprint implementors

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

### Backlog Items for Future Sprints

- **maxRayDepth** — Configurable ray recursion depth for the OptiX renderer. Currently there is
  no per-ray bounce limit; adding this would allow scenes to trade render quality for speed and
  may unblock the glass sponge skin darkness bug (manual test 53, see
  `CODE_IMPROVEMENTS.md:H-glass-sponge-skin-diffuse`). DSL stub added in Sprint 17 task 17.4;
  backend implementation required in `optix-jni` shaders and `RenderConfig`. Consider adding to
  Sprint 19 or as a standalone task once Sprint 18 GPU infrastructure is complete.
Content originally in Sprint 20 (GPU 4D Infrastructure) has moved here as 18.3.
