# Sprint 19: Advanced Geometry

**Sprint:** 19 - Advanced Geometry
**Status:** Not Started
**Estimate:** ~16.5 hours
**Branch:** `feature/sprint-19`
**Dependencies:** Sprint 18 (multi-GAS IAS, IS programs, GPU 4D math)

---

## Goal

Expand the geometry system with additional polytopes in 3D and 4D, analytical primitives,
planes as first-class geometry, a coordinate cross visualization, and a geometry registry
for extensibility.

## Success Criteria

- [ ] Octahedron, dodecahedron, icosahedron available as 3D primitives
- [ ] 16-cell, 24-cell, and 600-cell available as 4D primitives (standalone, no subdivision)
- [ ] Cone and torus available as analytical primitives (IS programs from Sprint 18)
- [ ] Planes are first-class scene geometry with materials (not miss shader)
- [ ] Coordinate cross / axis visualization works alongside other geometry
- [ ] Geometry registry: adding a new type requires registration, not engine modification
- [ ] All tests pass

---

## Tasks

### Task 19.1: Additional Polytopes in 3D

**Estimate:** 3h

Add octahedron, dodecahedron, and icosahedron as first-class primitives in the DSL and
rendering engine, alongside the existing cube/sphere/tetrahedron.

Implemented as triangle meshes via the existing `Builder` trait.

---

### Task 19.2: Additional Polytopes in 4D

**Estimate:** 4h
**Depends on:** Sprint 18.3 (GPU 4D math)

Add 16-cell, 24-cell, and 600-cell as 4D primitives. These are rendered using the GPU 4D
math from Sprint 18.3, avoiding the legacy CPU projection path.

**Scope:** Standalone polytopes only (rotate, project, view). Fractal subdivision on these
bases is deferred to a later sprint.

**Note:** The 600-cell has 120 vertices and 600 tetrahedral cells. Performance is expected
to be acceptable for standalone rendering; a level guard will be added if these are later
used as subdivision bases.

---

### Task 19.3: Analytical Primitives — Cone, Torus

**Estimate:** 2h
**Depends on:** Sprint 18.2 (IS program infrastructure)

Add cone and torus as analytical primitives using custom OptiX intersection programs.
Coexists with tessellated versions (via `ParametricTessellator`) — the analytical versions
provide exact intersection without tessellation artifacts.

DSL usage:
```scala
scene(
  cone(height = 2.0, radius = 1.0, material = gold),
  torus(majorRadius = 2.0, minorRadius = 0.5, material = glass)
)
```

Tessellated versions remain available as `TessellatedCone`, `TessellatedTorus`.

---

### Task 19.4: Planes as First-Class Geometry

**Estimate:** 2h
**Depends on:** Sprint 18.1 (multi-GAS IAS), Sprint 18.2 (IS programs)

Promote planes from the miss shader to proper scene geometry with:
- Custom IS program for plane intersection
- Per-plane material (inheritable via scene graph)
- Position and normal as parameters
- Zero or more planes per scene

The existing miss shader becomes a solid color or environment map background (Sprint 20).

---

### Task 19.5: Coordinate Cross (Axis Visualization)

**Estimate:** 1.5h
**Depends on:** Sprint 18.1 (multi-GAS IAS), Sprint 18.2 (IS programs)

Render XYZ axis lines for debugging scene layout and camera positioning.

- Thin analytical cylinders along X (red), Y (green), Z (blue) axes
- Configurable length and thickness
- Toggle on/off via CLI or keyboard shortcut

---

### Task 19.6: Geometry Registry

**Estimate:** 2h
**Depends on:** Sprint 17.3 (scene graph)

Replace the match expression in `OptiXEngine.selectMeshBuilder()` with a registration-based
`GeometryRegistry`. Adding a new geometry type means registering it (name, builder factory,
IS program if analytical, DSL constructor), not modifying the engine.

Scene graph nodes reference geometry by registered type. The registry provides:
- Name → builder/IS program mapping
- CLI option generation for `--objects` syntax
- DSL constructor availability

---

### Task 19.7: Documentation

**Estimate:** 2h

- Sprint retrospective
- CHANGELOG.md update
- `docs/guide/user-guide.md` geometry section updated
- Example scenes for each new primitive
- arc42 update: geometry registry architecture

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 19.1 | Additional polytopes in 3D | 3h | None |
| 19.2 | Additional polytopes in 4D | 4h | Sprint 18.3 |
| 19.3 | Analytical primitives: cone, torus | 2h | Sprint 18.2 |
| 19.4 | Planes as first-class geometry | 2h | Sprint 18.1, 18.2 |
| 19.5 | Coordinate cross / axis visualization | 1.5h | Sprint 18.1, 18.2 |
| 19.6 | Geometry registry | 2h | Sprint 17.3 |
| 19.7 | Documentation | 2h | All |
| **Total** | | **~16.5h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] `docs/guide/user-guide.md` geometry section updated
- [ ] Example scenes created for each new primitive
- [ ] Geometry registry documented for contributors

---

## Notes

### Formerly

This sprint combines content from the original Sprint 18 (Advanced Geometry) with
planes as first-class geometry and the geometry registry. The original Sprint 18 content
(multi-GAS IAS, IS infrastructure, GPU 4D math) has moved to the new Sprint 18 as
infrastructure prerequisites.

### Deferred

- **Fractal subdivision on 4D polychora** — deferred to Sprint 21+ (standalone rendering first)
- **Sponge cutaways** — remains in backlog (clipping-plane infrastructure non-trivial)
- **Schläfli polytope generator** — backlog (algorithmic construction from `{p,q}` symbols)
- **Cylinder as DSL primitive** — TBD based on whether the analytical cylinder IS program
  from the coordinate cross covers the use case
