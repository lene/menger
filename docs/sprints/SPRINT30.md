# Sprint 30: 4D Geometry II

**Sprint:** 30 - 4D Geometry II
**Status:** Not Started
**Estimate:** ~20 hours
**Branch:** `feature/sprint-30`
**Dependencies:** None (Sprint 18 GPU 4D math is already in place)

---

## Goal

Add 4D parametric surfaces (`f(u,v) → Vec4`) rendered via GPU projection, and 3D
parametric surface specializations (spherical harmonics, spherical coordinates). Extends
the existing 4D geometry infrastructure from Sprint 18.

---

## Success Criteria

- [ ] `addParametricSurface4D { fn = "(u,v) => cliffordTorus(u, v)" }` renders correctly
- [ ] Spherical harmonics displacement maps work on sphere geometry
- [ ] Both are DSL-accessible and integration-tested

---

## Tasks

### Task 30.1: 4D Parametric Surfaces `f(u,v) → Vec4` on GPU

**Estimate:** 8h

Evaluate `(u,v) → (x,y,z,w)` on GPU, project to 3D using existing `Project4D`, produce
triangle mesh for OptiX.

**Examples:**
- Clifford torus: `(cos(u), sin(u), cos(v), sin(v)) / sqrt(2)`
- Hypersphere patch: `(sin(u)cos(v), sin(u)sin(v), cos(u)cos(v2), cos(u)sin(v2))`

**Implementation:**
- CUDA kernel: evaluate `f(u,v)` on `N×N` grid → Vec4 array
- Apply `Project4D` rotation + projection to 3D
- Build triangle mesh from projected grid, upload to OptiX
- Re-evaluate on animation frame if `rotation4d` changes (same as Sprint 18 mesh update)

**DSL:**
```scala
addParametricSurface4D {
  fn = ParametricSurface4D.CliffordTorus
  uSteps = 64; vSteps = 64
  rotation4d = Rotation4D.identity
  material { ... }
}
```

---

### Task 30.2: Parametric Surface Specializations (3D)

**Estimate:** 5h

Two specializations of existing 3D parametric surfaces:

1. **Spherical coordinates** — `f(θ,φ) → r` as displacement on unit sphere, where `r`
   can be a mathematical expression (creates non-spherical surfaces)

2. **Spherical harmonics** — `Y_l^m(θ,φ)` as displacement; DSL selects `l` and `m`.
   Produces the classic lobe shapes used in physics visualization.

**DSL:**
```scala
addSphericalHarmonicSurface {
  l = 3; m = 2        // degree and order
  scale = 0.5f        // displacement scale
  colormap = Colormap.Plasma  // color by value
  resolution = 128
}
```

---

### Task 30.3: DSL Integration

**Estimate:** 4h

- `ParametricSurface4D` case class + `Scene` field
- `SphericalHarmonicSurface` case class + `Scene` field
- `SceneConfigurator` wiring for both new types
- JNI calls to upload mesh from GPU evaluation

---

### Task 30.4: Tests + Documentation

**Estimate:** 3h

- Integration tests: Clifford torus render, Y_3^2 spherical harmonic
- User guide: 4D Parametric Surfaces + Spherical Harmonics sections
- CHANGELOG.md entry

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 30.1 | 4D parametric surfaces f(u,v)→Vec4 on GPU | 8h |
| 30.2 | Parametric surface specializations (3D) | 5h |
| 30.3 | DSL integration | 4h |
| 30.4 | Tests + documentation | 3h |
| **Total** | | **~20h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
