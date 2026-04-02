# Sprint 21: Higher-Dimensional Fractals

**Sprint:** 21 - Higher-Dimensional Fractals
**Status:** Not Started
**Estimate:** ~14 hours
**Branch:** `feature/sprint-21`
**Dependencies:** Sprint 18 (18.3 — GPU 4D math), Sprint 19 (4D polychora as standalone)

---

## Goal

Implement Menger sponge and Sierpinski tetrahedron analogs in 4D and higher dimensions,
using the GPU 4D math infrastructure from Sprint 18.

## Success Criteria

- [ ] 4D Menger sponge analog renders correctly via GPU 4D math
- [ ] Higher-dimensional Sierpinski tetrahedron analogs work
- [ ] Interactive parameter space exploration (1D, using existing `scene(t)` system)
- [ ] All tests pass

---

## Tasks

### Task 21.1: 4D Menger Sponge Analog

**Estimate:** 5h
**Depends on:** Sprint 18.3 (GPU 4D math)

A 4D analog of the Menger sponge, using the GPU 4D transform infrastructure.
Rendered as a 3D cross-section at varying 4D rotation angles.

---

### Task 21.2: Higher-Dimensional Sierpinski Tetrahedron Analogs

**Estimate:** 4h
**Depends on:** Sprint 18.3 (GPU 4D math)

Sierpinski tetrahedron analogs in 4D (and higher if feasible), using the same GPU
infrastructure as 21.1.

---

### Task 21.3: Interactive Parameter Space Exploration

**Estimate:** 3h
**Depends on:** 21.1, 21.2

Allow real-time or animation-based exploration of fractal parameters:
- Fractal level (depth)
- 4D rotation angle(s)
- Cross-section position

Scope: 1D exploration using the existing `scene(t)` animation system. Map fractal
parameters to t for sweep animations.

**Deferred:** Multi-dimensional parameter exploration (independently varying 2-3+
parameters) is in the backlog.

---

### Task 21.4: Documentation

**Estimate:** 2h

- Sprint retrospective
- CHANGELOG.md update
- User guide section on higher-dimensional fractal analogs
- Example renders of 4D fractal cross-sections

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 21.1 | 4D Menger sponge analog | 5h | Sprint 18.3 |
| 21.2 | Higher-dim Sierpinski analogs | 4h | Sprint 18.3 |
| 21.3 | Interactive parameter exploration (1D) | 3h | 21.1, 21.2 |
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

### GPU 4D Path

All 4D geometry in this sprint uses the GPU 4D math from Sprint 18.3.
The CPU-side 4D pipeline (`Mesh4D`, `RotatedProjection`) is legacy at this point
and should not be used for new geometry.

### Deferred

- **Multi-dimensional parameter exploration** — backlog (independently varying multiple
  fractal parameters; needs parameter panel or multi-axis input mapping)
- **Fractal subdivision on 4D polychora** (16-cell, 24-cell, 600-cell as subdivision
  bases) — can be added in Sprint 22+ now that standalone polychora exist from Sprint 19
- **L-systems in 3D and 4D** — backlog
- **Rotopes** — backlog
