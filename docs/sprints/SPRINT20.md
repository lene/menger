# Sprint 20: GPU 4D Infrastructure

**Sprint:** 20 - GPU 4D Infrastructure
**Status:** Not Started
**Estimate:** ~11 hours
**Branch:** `feature/sprint-20`
**Dependencies:** Sprint 18 (15.2 — parametrized surfaces in 3D)

---

## Goal

Implement GPU-side 4D math as the foundation for Sprint 21's higher-dimensional fractal
analogs. Add parametrized 4D surfaces.

## Success Criteria

- [ ] CUDA kernel for 4D transform and projection (GPU-side 4D math)
- [ ] Parametrized surfaces in 4D via `f(u, v) → Vec4` pipeline
- [ ] All tests pass

---

## Tasks

### Task 20.1: CUDA 4D Transform and Projection

**Estimate:** 5h

Move 4D rotation, projection, and coordinate transforms to GPU-side CUDA code.
This is the core prerequisite for Sprint 21 (4D Menger and Sierpinski analogs).

Current state: 4D transforms computed on CPU and passed as geometry to OptiX.
Target state: 4D transforms evaluated per-ray on the GPU for procedural geometry.

---

### Task 20.2: Parametrized Surfaces in 4D

**Estimate:** 4h

Extend the 3D parametrized surface infrastructure (15.2) to 4D:
- `f(u, v) → Vec4` → project to 3D for rendering
- Examples: 4D torus, Clifford torus, hypersphere patches

**Depends on:** 15.2 (parametrized surfaces in 3D) + 20.1 (GPU 4D math)

---

### Task 20.3: Documentation

**Estimate:** 2h

Sprint retrospective, CHANGELOG.md update, and developer docs for the GPU 4D math API.

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 20.1 | CUDA 4D transform + projection | 5h | None |
| 20.2 | Parametrized 4D surfaces | 4h | 15.2, 20.1 |
| 20.3 | Documentation | 2h | All |
| **Total** | | **~11h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] GPU 4D math API documented for Sprint 21 implementors

---

## Notes

Task 20.1 is the critical path for Sprint 21. The GPU 4D math infrastructure it creates
enables the higher-dimensional Menger and Sierpinski analogs planned for Sprint 21.

Project website with feedback button moved to Sprint 16 (Developer Infrastructure & Website).
