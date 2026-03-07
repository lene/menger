# Sprint 18: GPU 4D & Parametrized 4D Surfaces

**Sprint:** 18 - GPU 4D & Parametrized 4D Surfaces
**Status:** Not Started
**Estimate:** ~14 hours
**Branch:** `feature/sprint-18`
**Dependencies:** Sprint 16 (16.4 — parametrized surfaces in 3D)

---

## Goal

Implement GPU-side 4D math as the foundation for Sprint 19's higher-dimensional fractal
analogs. Add parametrized 4D surfaces and a project website.

## Success Criteria

- [ ] CUDA kernel for 4D transform and projection (GPU-side 4D math)
- [ ] Parametrized surfaces in 4D via `f(u, v) → Vec4` pipeline
- [ ] Project website with GitHub/GitLab feedback button
- [ ] All tests pass

---

## Tasks

### Task 18.1: CUDA 4D Transform and Projection

**Estimate:** 5h

Move 4D rotation, projection, and coordinate transforms to GPU-side CUDA code.
This is the core prerequisite for Sprint 19 (4D Menger and Sierpinski analogs).

Current state: 4D transforms computed on CPU and passed as geometry to OptiX.
Target state: 4D transforms evaluated per-ray on the GPU for procedural geometry.

---

### Task 18.2: Parametrized Surfaces in 4D

**Estimate:** 4h

Extend the 3D parametrized surface infrastructure (16.4) to 4D:
- `f(u, v) → Vec4` → project to 3D for rendering
- Examples: 4D torus, Clifford torus, hypersphere patches

**Depends on:** 16.4 (parametrized surfaces in 3D) + 18.1 (GPU 4D math)

---

### Task 18.3: Website with Feedback Button

**Estimate:** 3h

Create a project website (static, hosted on GitLab Pages or GitHub Pages) with:
- Project overview and example renders
- Feedback button that opens a pre-filled GitHub/GitLab issue

---

### Task 18.4: Documentation

**Estimate:** 2h

Sprint retrospective, CHANGELOG.md update, and developer docs for the GPU 4D math API.

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 18.1 | CUDA 4D transform + projection | 5h | None |
| 18.2 | Parametrized 4D surfaces | 4h | 16.4, 18.1 |
| 18.3 | Website with feedback button | 3h | None |
| 18.4 | Documentation | 2h | All |
| **Total** | | **~14h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] GPU 4D math API documented for Sprint 19 implementors

---

## Notes

Task 18.1 is the critical path for Sprint 19. The GPU 4D math infrastructure it creates
enables the higher-dimensional Menger and Sierpinski analogs planned for Sprint 19.
