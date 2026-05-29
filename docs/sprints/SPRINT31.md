# Sprint 31: Advanced Geometry

**Sprint:** 31 - Advanced Geometry
**Status:** Not Started
**Estimate:** ~27 hours
**Branch:** `feature/sprint-31`
**Dependencies:** None

---

## Goal

Three advanced geometry features: sponge cross-sections via clipping planes, Schläfli
polytope generation from `{p,q}` symbols, and fractal subdivision using polychora
(16-cell, 24-cell, 600-cell) as bases instead of the cube.

---

## Success Criteria

- [ ] `clipPlane { normal = Vec3(1,0,0), distance = 0.5 }` clips any geometry revealing cross-section
- [ ] `addPolytope { schlafli = "{3,5}" }` generates an icosahedron mesh
- [ ] `addFractal4D { base = Polychoron.Cell600, level = 2 }` renders correctly
- [ ] All features integration-tested

---

## Tasks

### Task 31.1: Sponge Cutaways via Clipping Planes

**Estimate:** 6h

Clip any geometry along one or more planes to reveal internal cross-section structure.
Particularly useful for visualizing the interior of Menger sponges and 4D fractals.

**DSL:**
```scala
scene {
  clipPlanes = List(
    ClipPlane(normal = Vec3(1, 0, 0), distance = 0.0f)  // clip at x=0
  )
}
```

**Implementation:**
- In raygen or hit shader: after hit, check if hit point is on the clipped side of any
  plane; if so, treat as miss (or recurse for interior surface)
- Pass up to 4 clip planes as `Params` fields (already have `PlaneParams` pattern)
- Interior surface: either show the cross-section color or use a fixed "cut" material

---

### Task 31.2: Schläfli Polytope Generator

**Estimate:** 10h

Algorithmically construct regular polytopes from their Schläfli symbol `{p,q}` (3D)
or `{p,q,r}` (4D projected to 3D), producing a triangle mesh.

**Supported symbols (3D):**
- `{3,3}` tetrahedron, `{4,3}` cube, `{3,4}` octahedron
- `{5,3}` dodecahedron, `{3,5}` icosahedron

**Supported symbols (4D, project to 3D):**
- `{3,3,3}` 5-cell, `{4,3,3}` 8-cell (tesseract), `{3,3,4}` 16-cell
- `{3,4,3}` 24-cell, `{5,3,3}` 120-cell, `{3,3,5}` 600-cell

**DSL:**
```scala
addPolytope {
  schlafli = "{3,5}"    // icosahedron
  scale = 1.0f
  material { ... }
}
```

**Implementation:**
- Wythoff construction for 3D: reflection group algorithm
- 4D: generate vertices via root systems, project to 3D using existing Project4D
- Output: triangle mesh → existing `setTriangleMesh` pipeline

---

### Task 31.3: Fractal Subdivision on Polychora

**Estimate:** 8h

Use 16-cell, 24-cell, or 600-cell as the subdivision base for IFS fractals instead of
the cube (current Menger4D base). Enables new fractal shapes with different symmetry.

**DSL:**
```scala
addFractal4D {
  base = FractalBase4D.Cell16    // or Cell24, Cell600
  level = 2
  rotation4d = Rotation4D.identity
  material { ... }
}
```

**Implementation:**
- Define IFS rules for each polychoron base (contractive affine maps preserving the
  symmetry group)
- Implement new intersection shaders `hit_cell16_fractal4d.cu`, etc.
- Reuse `Menger4DData` pattern from `menger-geometry` (MengerParams extension)

---

### Task 31.4: Tests + Documentation

**Estimate:** 3h

- Integration tests: clip plane on Menger sponge, icosahedron, 16-cell fractal
- User guide: Clipping Planes, Schläfli Generator, Polychora Fractals sections
- CHANGELOG.md entry

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 31.1 | Sponge cutaways via clipping planes | 6h |
| 31.2 | Schläfli polytope generator | 10h |
| 31.3 | Fractal subdivision on polychora | 8h |
| 31.4 | Tests + documentation | 3h |
| **Total** | | **~27h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
