# Sprint 18: Advanced Geometry

**Sprint:** 18 - Advanced Geometry
**Status:** Not Started
**Estimate:** ~15.5 hours
**Branch:** `feature/sprint-18`
**Dependencies:** Sprint 15 (15.3 — cylinder/cone/torus primitives, for 18.3 surface infra)

> **Note:** 4D and 3D sponge cutaways (originally 16.1) have been moved to the backlog.
> They require significant clipping-plane infrastructure with uncertain scope.

---

## Goal

Expand the geometry system with additional polytopes in 3D and 4D, parametrized surfaces,
a coordinate cross visualization, and geometry documentation.

## Success Criteria

- [ ] Octahedron, dodecahedron, icosahedron available as primitives
- [ ] 16-cell, 24-cell, and 600-cell available as 4D primitives
- [ ] Parametrized surfaces in 3D (sphere patches, tori, implicit surfaces) work
- [ ] Coordinate cross / axis visualization available via CLI
- [ ] User guide geometry section updated
- [ ] All tests pass

---

## Tasks

### Task 18.1: Additional Polytopes in 3D

**Estimate:** 3h

Add octahedron, dodecahedron, and icosahedron as first-class primitives in the DSL and
rendering engine, alongside the existing cube/sphere/tetrahedron.

---

### Task 18.2: Additional Polytopes in 4D

**Estimate:** 4h

Add 16-cell, 24-cell, and 600-cell as 4D primitives. These are the 4D analogs of the
octahedron, cuboctahedron, and icosahedron.

---

### Task 18.3: Parametrized Surfaces in 3D

**Estimate:** 4h

Infrastructure for rendering parametrized surfaces defined by `f(u, v) → Vec3`.
Initial implementations: sphere patches, tori, implicit surfaces.

This is a prerequisite for Sprint 20's parametrized 4D surfaces (20.2).

---

### Task 18.4: Coordinate Cross (Axis Visualization)

**Estimate:** 1.5h

Render XYZ axis lines for debugging scene layout and camera positioning.

#### CLI

```bash
menger --optix --show-axes          # Show coordinate cross at origin
menger --optix --show-axes --axis-length 5.0  # Custom axis length
```

#### Implementation

- Render thin cylinders along X (red), Y (green), Z (blue) axes
- Configurable length and thickness
- Toggle on/off via CLI or keyboard shortcut

---

### Task 18.5: User Guide — Geometry Section

**Estimate:** 1h

Update USER_GUIDE.md with documentation for all new geometry primitives and the
parametrized surface API.

---

### Task 18.6: Documentation

**Estimate:** 2h

Sprint retrospective, CHANGELOG.md update, and example scenes for new geometry.

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 18.1 | Additional polytopes in 3D | 3h | None |
| 18.2 | Additional polytopes in 4D | 4h | None |
| 18.3 | Parametrized surfaces in 3D | 4h | None |
| 18.4 | Coordinate cross / axis visualization | 1.5h | None |
| 18.5 | User guide: geometry | 1h | 18.1–18.4 |
| 18.6 | Documentation | 2h | All |
| **Total** | | **~15.5h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] USER_GUIDE.md geometry section updated
- [ ] Example scenes created for each new primitive

---

## Backlog Note

**Sponge cutaways** (clipping-plane cross-section views of 3D and 4D Menger sponges) were
originally Sprint 16.1. Moved to backlog due to uncertain scope (clipping-plane geometry
infrastructure is non-trivial and not a prerequisite for other planned work).
