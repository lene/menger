# Sprint 21: Higher-Dimensional Fractals

**Sprint:** 21 - Higher-Dimensional Fractals
**Status:** Not Started
**Estimate:** ~16 hours
**Branch:** `feature/sprint-21`
**Dependencies:** Sprint 20 (20.1 — GPU 4D math)

> **Note:** 3D Menger cutaway tools (originally 19.3, depends on sponge cutaways) have been
> removed from this sprint. Sponge cutaways were moved to the backlog — see SPRINT18.md.

---

## Goal

Implement Menger sponge and Sierpinski tetrahedron analogs in 4D and higher dimensions.
Promoted from long-term backlog after the GPU 4D infrastructure in Sprint 20 makes this
tractable.

## Success Criteria

- [ ] 4D Menger sponge analog renders correctly via GPU 4D math
- [ ] Higher-dimensional Sierpinski tetrahedron analogs work
- [ ] Interactive parameter space exploration for fractal parameters
- [ ] All tests pass

---

## Tasks

### Task 21.1: 4D Menger Sponge Analog

**Estimate:** 5h

A 4D analog of the Menger sponge, using the GPU 4D transform infrastructure from 20.1.
Rendered as a 3D cross-section at varying 4D rotation angles.

**Depends on:** 20.1 (GPU 4D math)

---

### Task 21.2: Higher-Dimensional Sierpinski Tetrahedron Analogs

**Estimate:** 4h

Sierpinski tetrahedron analogs in 4D (and higher if feasible), using the same GPU
infrastructure as 21.1.

**Depends on:** 20.1 (GPU 4D math)

---

### Task 21.3: Interactive Parameter Space Exploration

**Estimate:** 3h

Allow real-time or animation-based exploration of fractal parameters:
- Fractal level (depth)
- 4D rotation angle(s)
- Cross-section position

Using the existing `scene(t)` animation system.

**Depends on:** 21.1, 21.2

---

### Task 21.4: Documentation

**Estimate:** 2h

Sprint retrospective, CHANGELOG.md update, and user guide section on higher-dimensional
fractal analogs.

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 21.1 | 4D Menger sponge analog | 5h | 20.1 |
| 21.2 | Higher-dim Sierpinski analogs | 4h | 20.1 |
| 21.3 | Interactive parameter exploration | 3h | 21.1, 21.2 |
| 21.4 | Documentation | 2h | All |
| **Total** | | **~14h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] Example renders of 4D fractal cross-sections

---

## Notes

### Long-Term Backlog (Sprint 22+)

Items that depend on this sprint or extend it further:
- L-systems in 3D and 4D
- Rotopes (higher-dimensional geometry via rotation of lower-dimensional shapes)
- Stereoscopic 3D rendering
- 3D Menger cutaway visualization (depends on sponge cutaways — currently in backlog)
