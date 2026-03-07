# Sprint 16: Advanced Geometry

**Sprint:** 16 - Advanced Geometry
**Status:** Not Started
**Estimate:** ~17 hours
**Branch:** `feature/sprint-16`
**Dependencies:** None (16.1-16.4 are independent)

---

## Goal

Expand the geometry system with 4D cutaways, additional polytopes in 3D and 4D,
and parametrized surfaces. Establishes the surface infrastructure needed for Sprint 18.

## Success Criteria

- [ ] 4D and 3D sponge cutaways via clipping planes work
- [ ] Octahedron, dodecahedron, icosahedron available as primitives
- [ ] 16-cell, 24-cell, and 600-cell available as 4D primitives
- [ ] Parametrized surfaces in 3D (sphere patches, tori, implicit surfaces) work
- [ ] User guide geometry section updated
- [ ] All tests pass

---

## Tasks

### Task 16.1: 4D and 3D Sponge Cutaways

**Estimate:** 3h

Implement clipping-plane geometry to create cross-section views of 3D and 4D Menger
sponges. Standalone feature; no dependencies.

---

### Task 16.2: Additional Polytopes in 3D

**Estimate:** 3h

Add octahedron, dodecahedron, and icosahedron as first-class primitives in the DSL and
rendering engine, alongside the existing cube/sphere/tetrahedron.

---

### Task 16.3: Additional Polytopes in 4D

**Estimate:** 4h

Add 16-cell, 24-cell, and 600-cell as 4D primitives. These are the 4D analogs of the
octahedron, cuboctahedron, and icosahedron.

---

### Task 16.4: Parametrized Surfaces in 3D

**Estimate:** 4h

Infrastructure for rendering parametrized surfaces defined by `f(u, v) → Vec3`.
Initial implementations: sphere patches, tori, implicit surfaces.

This is a prerequisite for Sprint 18's parametrized 4D surfaces (18.2).

---

### Task 16.5: User Guide — Geometry Section

**Estimate:** 1h

Update USER_GUIDE.md with documentation for all new geometry primitives and the
parametrized surface API.

---

### Task 16.6: Documentation

**Estimate:** 2h

Sprint retrospective, CHANGELOG.md update, and example scenes for new geometry.

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 16.1 | 4D and 3D sponge cutaways | 3h | None |
| 16.2 | Polytopes in 3D | 3h | None |
| 16.3 | Polytopes in 4D | 4h | None |
| 16.4 | Parametrized surfaces in 3D | 4h | None |
| 16.5 | User guide: geometry | 1h | 16.1–16.4 |
| 16.6 | Documentation | 2h | All |
| **Total** | | **~17h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] USER_GUIDE.md geometry section updated
- [ ] Example scenes created for each new primitive
