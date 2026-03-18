# Sprint 16: Developer Infrastructure & Website

**Sprint:** 16 - Developer Infrastructure & Website
**Status:** Not Started
**Estimate:** ~11 hours
**Branch:** `feature/sprint-16`
**Dependencies:** None

---

## Goal

Improve developer experience through pre-push hook optimisation, documentation refresh,
CI improvements, a project website, and CUDA version compatibility testing.

## Success Criteria

- [ ] Pre-push hook parallelized with measurable wall-clock improvement
- [ ] AGENTS.md and developer documentation refreshed
- [ ] Test coverage improved; Valgrind CI added
- [ ] Project website live with GitHub/GitLab feedback button
- [ ] Full test suite verified on CUDA 12 and CUDA 13 via CI Docker images
- [ ] All tests pass

---

## Tasks

### Task 16.1: Optimise Pre-Push Hook

**Estimate:** 2h

Analyze pre-push hook bottlenecks and parallelize where possible. Target: measurable
reduction in wall-clock time for a typical push.

#### Approach

- Profile current hook execution to identify slowest steps
- Parallelize independent checks (compile, lint, test suites)
- Document the parallelization approach for future maintainers

---

### Task 16.2: Developer Documentation & AGENTS.md Refresh

**Estimate:** 2h

Update AGENTS.md with better instructions for:
- Updating documentation and changelog
- Monitoring CI pipelines after push
- Using `glab` for GitLab operations
- Sprint lifecycle and commit conventions

---

### Task 16.3: Test Coverage + Valgrind CI Improvements

**Estimate:** 2h

Analyze current test coverage, add tests where coverage is thin, and add Valgrind
standalone test verification to CI pipeline.

---

### Task 16.4: Project Website with Feedback Button

**Estimate:** 3h

Create a project website (static, hosted on GitLab Pages or GitHub Pages) with:
- Project overview and example renders
- Feedback button that opens a pre-filled GitHub/GitLab issue

---

### Task 16.5: Test on CUDA 12 and 13

**Estimate:** 2h

Add CI Docker images for CUDA 12 and CUDA 13, run the full test suite on both.
Document any version-specific workarounds.

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 16.1 | Optimise pre-push hook | 2h | None |
| 16.2 | Developer docs + AGENTS.md refresh | 2h | None |
| 16.3 | Test coverage + Valgrind CI | 2h | None |
| 16.4 | Project website with feedback button | 3h | None |
| 16.5 | Test on CUDA 12 + 13 | 2h | None |
| **Total** | | **~11h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing (including CUDA 12 + 13)
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] Developer documentation updated
- [ ] Website deployed and accessible
