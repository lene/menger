# Sprint 15: DSL Completeness & Infrastructure

**Sprint:** 15 - DSL Completeness & Infrastructure
**Status:** Not Started
**Estimate:** ~18 hours
**Branch:** `feature/sprint-15`
**Dependencies:** Sprint 10 (Scala DSL), Sprint 14 (Video Output)

---

## Goal

Complete deferred DSL features from Sprint 10 and address infrastructure/polish work.
Plain Scala in `scene(t)` already serves as the animation DSL — this sprint adds
convenience helpers, tooling, and CI improvements.

## Success Criteria

- [ ] DSL supports setting window width, height, saveName, and headless mode
- [ ] Scene composition helpers available in DSL (SceneBuilder utilities)
- [ ] Procedural placement helpers available in DSL
- [ ] Bezier/spline camera path utility implemented as pure Scala helper
- [ ] Runtime DSL scene evaluation works (not just compile-time)
- [ ] Animation export/import (JSON format for t-param frame configs)
- [ ] Pre-push hook parallelized (measurable speed improvement)
- [ ] Developer and agent documentation updated
- [ ] Test coverage improved; Valgrind CI added

---

## Tasks

### Task 15.1: DSL Window/Output Settings

**Estimate:** 2h

Add `width`, `height`, `saveName`, and `headless` to the DSL configuration API (currently CLI-only).

---

### Task 15.2: Scene Composition Helpers

**Estimate:** 2h

Add `SceneBuilder` utilities for common scene construction patterns (grouping, instancing, positioning).

---

### Task 15.3: Procedural Placement Helpers in DSL

**Estimate:** 2h

Add helpers for procedural object placement (grids, rings, spirals, random distributions).

---

### Task 15.4: Bezier/Spline Camera Path Utility

**Estimate:** 2h

Pure Scala helper (no engine changes) for building smooth camera paths as a function of `t`.
Useful with the existing `scene(t)` animation system.

---

### Task 15.5: Runtime DSL Scene Evaluation

**Estimate:** 2h

Allow scenes to be evaluated at runtime (not just compiled in). Enables hot-reload and
interactive iteration without recompiling.

---

### Task 15.6: Animation Export/Import (JSON)

**Estimate:** 2h

JSON format for saving/loading t-parameter frame configurations (frame count, range, output paths).
Enables repeatable animation runs from config files.

---

### Task 15.7: Optimize Pre-Push Hook

**Estimate:** 2h

Analyze pre-push hook bottlenecks and parallelize where possible. Target: measurable
reduction in wall-clock time for a typical push.

---

### Task 15.8: Developer Documentation & Agent Instructions

**Estimate:** 1h

Update AGENTS.md with better instructions for:
- Updating documentation and changelog
- Monitoring CI pipelines after push
- Using `glab` for GitLab operations

---

### Task 15.9: Test Coverage + Valgrind CI

**Estimate:** 1h

Analyze current test coverage, add tests where coverage is thin, and add Valgrind
standalone test verification to CI.

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 15.1 | DSL window/output settings | 2h |
| 15.2 | Scene composition helpers | 2h |
| 15.3 | Procedural placement helpers | 2h |
| 15.4 | Bezier/spline camera path utility | 2h |
| 15.5 | Runtime DSL evaluation | 2h |
| 15.6 | Animation export/import (JSON) | 2h |
| 15.7 | Optimize pre-push hook | 2h |
| 15.8 | Developer docs + agent instructions | 1h |
| 15.9 | Test coverage + Valgrind CI | 1h |
| **Total** | | **~18h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] Developer documentation updated

---

## Notes

### Background

Sprint 10 (Scala DSL) deferred several features to avoid scope creep. This sprint
completes that deferred work, combined with infrastructure improvements that have
accumulated since Sprint 11.

The t-parameter animation system (Sprint 12) eliminated the need for a separate
animation DSL — `scene(t)` is the DSL.
