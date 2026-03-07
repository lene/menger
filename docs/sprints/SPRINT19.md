# Sprint 19: Higher-Dimensional Fractal Analogs

**Sprint:** 19 - Higher-Dimensional Fractal Analogs
**Status:** Not Started
**Estimate:** ~16 hours
**Branch:** `feature/sprint-19`
**Dependencies:** Sprint 16 (16.1 — sponge cutaways), Sprint 18 (18.1 — GPU 4D math)

---

## Goal

Implement Menger sponge and Sierpinski tetrahedron analogs in 4D and higher dimensions.
Promoted from long-term backlog after the GPU 4D infrastructure in Sprint 18 makes this
tractable.

## Success Criteria

- [ ] 4D Menger sponge analog renders correctly via GPU 4D math
- [ ] Higher-dimensional Sierpinski tetrahedron analogs work
- [ ] 3D Menger cutaway visualization tools available (depends on 16.1)
- [ ] Interactive parameter space exploration for fractal parameters
- [ ] All tests pass

---

## Tasks

### Task 19.1: 4D Menger Sponge Analog

**Estimate:** 5h

A 4D analog of the Menger sponge, using the GPU 4D transform infrastructure from 18.1.
Rendered as a 3D cross-section at varying 4D rotation angles.

**Depends on:** 18.1 (GPU 4D math)

---

### Task 19.2: Higher-Dimensional Sierpinski Tetrahedron Analogs

**Estimate:** 4h

Sierpinski tetrahedron analogs in 4D (and higher if feasible), using the same GPU
infrastructure as 19.1.

**Depends on:** 18.1 (GPU 4D math)

---

### Task 19.3: 3D Menger Cutaway Visualization Tools

**Estimate:** 3h

User-facing tools for exploring 3D Menger sponge cross-sections via clipping planes,
building on the cutaway infrastructure from 16.1.

**Depends on:** 16.1 (sponge cutaways)

---

### Task 19.4: Interactive Parameter Space Exploration

**Estimate:** 2h

Allow real-time or animation-based exploration of fractal parameters:
- Fractal level (depth)
- 4D rotation angle(s)
- Cross-section position

Using the existing `scene(t)` animation system.

---

### Task 19.5: Documentation

**Estimate:** 2h

Sprint retrospective, CHANGELOG.md update, and user guide section on higher-dimensional
fractal analogs.

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 19.1 | 4D Menger sponge analog | 5h | 18.1 |
| 19.2 | Higher-dim Sierpinski analogs | 4h | 18.1 |
| 19.3 | 3D Menger cutaway tools | 3h | 16.1 |
| 19.4 | Interactive parameter exploration | 2h | 19.1, 19.2 |
| 19.5 | Documentation | 2h | All |
| **Total** | | **~16h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] Example renders of 4D fractal cross-sections

---

## Notes

### Long-Term Backlog (Sprint 20+)

Items that depend on this sprint or extend it further:
- L-systems in 3D and 4D
- Rotopes (higher-dimensional geometry via rotation of lower-dimensional shapes)
- Stereoscopic 3D rendering
