# Sprint 15: Visual Enhancements & Primitives

**Sprint:** 15 - Visual Enhancements & Primitives
**Status:** Not Started
**Estimate:** ~10.5 hours
**Branch:** `feature/sprint-15`
**Dependencies:** Sprint 14 (rendering correctness baseline)

---

## Goal

Add soft shadows with area lights, depth of field, and additional geometric primitives
(cylinder, cone, torus) to enrich scene composition and visual quality.

## Success Criteria

- [ ] Soft shadows with area lights (penumbra visible)
- [ ] Depth of field (camera aperture, bokeh)
- [ ] Additional primitives: cylinder, cone, torus available in DSL
- [ ] Documentation and examples updated
- [ ] All tests pass

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

#### Files to Modify

- `menger-app/src/main/scala/menger/dsl/Light.scala` — add `AreaLight` case
- `optix-jni/src/main/native/shaders/shadows.cu` — multi-sample shadow rays
- `optix-jni/src/main/native/include/OptiXData.h` — area light data structure

---

### Task 15.2: Depth of Field

**Estimate:** 3h

Camera aperture simulation producing bokeh (out-of-focus blur).

#### CLI

```bash
menger --optix --aperture 0.1 --focus-distance 5.0
```

#### Implementation

- Jitter camera ray origins within aperture disk
- Focus plane at specified distance
- Multiple samples per pixel (reuse AA infrastructure)
- Configurable aperture size and focus distance

#### Files to Modify

- `optix-jni/src/main/native/shaders/raygen_primary.cu` — add aperture jitter
- `menger-app/src/main/scala/menger/MengerCLIOptions.scala` — add `--aperture`, `--focus-distance`

---

### Task 15.3: Additional Primitives: Cylinder, Cone, Torus

**Estimate:** 3h

Add cylinder, cone, and torus geometry types for richer scene composition.

#### DSL Syntax

```scala
Scene(
  objects = List(
    Cylinder(pos = (0f, 0f, 0f), radius = 0.5f, height = 2f, material = Material.Chrome),
    Cone(pos = (2f, 0f, 0f), radius = 0.5f, height = 1.5f, material = Material.Gold),
    Torus(pos = (-2f, 0f, 0f), majorRadius = 1f, minorRadius = 0.3f, material = Material.Glass)
  ),
  // ...
)
```

#### Implementation

- Triangle mesh generation for each primitive
- Parametric UV coordinates for texturing
- Normal computation for smooth shading
- Integration with existing scene builder system

#### Files to Create

- `menger-app/src/main/scala/menger/dsl/Cylinder.scala` (or extend SceneObject)
- `menger-app/src/main/scala/menger/dsl/Cone.scala`
- `menger-app/src/main/scala/menger/dsl/Torus.scala`

---

### Task 15.4: Documentation & Examples

**Estimate:** 1.5h

Update USER_GUIDE.md and CHANGELOG.md for all new features.

#### Sections to Add

- Soft shadows examples with area lights
- Depth of field photography guide
- New primitive reference (cylinder, cone, torus)
- Example scenes showcasing each feature

---

## Summary

| Task | Description | Estimate | Priority |
|------|-------------|----------|----------|
| 15.1 | Soft shadows (area lights) | 3h | High |
| 15.2 | Depth of field | 3h | High |
| 15.3 | Additional primitives (cylinder, cone, torus) | 3h | High |
| 15.4 | Documentation & examples | 1.5h | High |
| **Total** | | **~10.5h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] USER_GUIDE.md updated
- [ ] Example scenes created and tested
