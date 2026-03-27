# Sprint 15: Visual Enhancements & Parametric Geometry

**Sprint:** 15 - Visual Enhancements & Parametric Geometry
**Status:** In Progress
**Estimate:** ~13 hours
**Branch:** `feature/sprint-15`
**Dependencies:** Sprint 14 (rendering correctness baseline, caustics PPM foundation)

---

## Goal

Add soft shadows with area lights, a general parametric surface infrastructure, and extend
caustics to work with complex geometry beyond spheres.

## Success Criteria

- [x] Soft shadows with area lights (penumbra visible)
- [x] Parametrized surfaces `f(u,v) → Vec3` renderable in DSL and OptiX
- [x] Caustics work correctly for parametric geometry (not just spheres)
- [x] Documentation and examples updated
- [x] All tests pass

---

## Tasks

### Task 15.1: Soft Shadows with Area Lights

**Estimate:** 3h

Replace point/directional lights with area lights that produce penumbra (soft shadow edges).

#### Implementation

- Add `AreaLight` type to DSL: `AreaLight(position, size, intensity, color)`
- Multiple shadow rays per pixel sampling the light area
- Configurable shadow samples (default: 4, max: 16)
- CLI: `--shadow-samples N`

#### Interaction with Colored Shadows

Area lights cast multiple shadow rays per pixel. Each ray independently accumulates
transparent attenuation through the existing anyhit program chain. The final shadow
contribution is the average of all per-ray attenuations. This means:

- Colored shadow tinting is preserved under area lights
- The anyhit accumulation path (`accumulateShadowAttenuation`) is unchanged
- Performance cost: O(shadow_samples) anyhit evaluations per lit surface point
- **Spec alignment required before implementation:** confirm averaging strategy
  (per-channel average vs. per-ray boolean then average) with user before starting.

#### Files to Modify

- `menger-app/src/main/scala/menger/dsl/Light.scala` — add `AreaLight` case
- `optix-jni/src/main/native/shaders/shadows.cu` — multi-sample shadow rays
- `optix-jni/src/main/native/include/OptiXData.h` — area light data structure

---

### Task 15.2: Parametrized Surfaces in 3D

**Estimate:** 4h

Infrastructure for rendering parametric surfaces defined by `f(u, v) → Vec3`.
Cylinder, cone, torus, sphere patches, and implicit surfaces can all be expressed
this way — no separate DSL types needed at this stage.

#### DSL Syntax

```scala
Scene(
  objects = List(
    ParametricSurface(
      f = (u, v) => Vec3(cos(u) * sin(v), sin(u) * sin(v), cos(v)),  // sphere
      uRange = (0f, 2f * Pi),
      vRange = (0f, Pi),
      uSteps = 64,
      vSteps = 32,
      material = Material.Glass
    )
  )
)
```

#### Implementation

- `f(u, v) → Vec3` evaluated on CPU, tessellated to triangle mesh
- UV coordinates passed through for future texturing
- Normal computation: cross product of partial derivatives
- Integration with existing triangle mesh / IAS scene builder

#### Files to Create / Modify

- `menger-app/src/main/scala/menger/dsl/ParametricSurface.scala`
- `menger-app/src/main/scala/menger/objects/ParametricSurfaceBuilder.scala`

**Spec alignment required before implementation:** agree on tessellation resolution API,
closed vs. open surfaces, and seam handling.

---

### Task 15.3: Caustics for General Geometry

**Estimate:** 4h

Extend caustics (PPM) beyond sphere-only refractive objects to work with parametric
surfaces and other triangle-mesh geometry.

#### Background

Sprint 14 caustics use sphere-specific refraction logic in `caustics_ppm.cu`. The photon
tracing pass hard-codes Snell's law for a sphere refractive object. General mesh geometry
requires surface normals from the triangle hit programs instead.

#### Implementation

- Replace sphere-specific refraction in photon tracing with generic normal-based Snell's law
- Photon hits on any refractive triangle mesh (e.g. parametric torus) should refract correctly
- Validate with a parametric surface from Task 15.2 as the refractive object

**Spec alignment required before implementation:** review the sphere-specific code paths
in `caustics_ppm.cu` together with the user before starting, to agree on scope and approach.

#### Files to Modify

- `optix-jni/src/main/native/shaders/caustics_ppm.cu` — generalize photon refraction
- `optix-jni/src/main/native/CausticsRenderer.cpp` — if hit-point generation needs updating

---

### Task 15.4: Documentation & Examples

**Estimate:** 2h

Update `docs/guide/user-guide.md`, `docs/guide/advanced.md`, and CHANGELOG.md for all new features.

#### Sections to Add

- Soft shadows with area lights (including note on colored shadow interaction)
- Parametric surface API reference with examples (sphere, torus, cylinder)
- Caustics — update Known Limitations to reflect general geometry support

---

## Summary

| Task | Description | Estimate | Priority |
|------|-------------|----------|----------|
| 15.1 | Soft shadows (area lights) | 3h | High |
| 15.2 | Parametrized surfaces `f(u,v)` | 4h | High |
| 15.3 | Caustics for general geometry | 4h | High |
| 15.4 | Documentation & examples | 2h | High |
| **Total** | | **~13h** | |

---

## Definition of Done

- [x] All success criteria met
- [x] All tests passing
- [x] Code quality checks pass: `sbt "scalafix --check"`
- [x] CHANGELOG.md updated
- [x] `docs/guide/` updated
- [x] Example scenes created and tested

---

## Deferred from This Sprint

- **Depth of field / aperture / bokeh** → Backlog (not urgent)
- **Cylinder, cone, torus as explicit DSL types** → delivered implicitly via Task 15.2

---

## Implementation Notes (Actual vs. Planned)

### Task 15.1: Shadow samples are per-light, not a global flag

The sprint plan specified `--shadow-samples N` as a top-level CLI flag. The actual
implementation embeds samples inside the `--light` spec:
```
--light area:px,py,pz:nx,ny,nz:radius[:samples[:intensity[:color]]]
```
This is more flexible (different lights can have different sample counts) and avoids
a global flag that would only apply to area lights.

### Task 15.2: ParametricSurface lives in SceneObject.scala, not a new file

The plan listed `dsl/ParametricSurface.scala` and `objects/ParametricSurfaceBuilder.scala`
as new files. In practice, `ParametricSurface` was added as a new `case class` inside the
existing `SceneObject.scala`, and tessellation lives in `ParametricTessellator.scala`
(not `ParametricSurfaceBuilder`).

### Task 15.3: Multi-light and multi-plane caustics still deferred

Photon emission only reads `params.lights[0]` and only deposits on `params.planes[0]`.
These are pre-existing TODOs in `caustics_ppm.cu` — not regressions from the generalization.
