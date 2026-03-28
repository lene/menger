# Sprint 16: Developer Infrastructure & Website

**Sprint:** 16 - Developer Infrastructure & Website
**Status:** In Progress
**Estimate:** ~16 hours
**Branch:** `feature/sprint-16`
**Dependencies:** None

---

## Goal

Improve developer experience through pre-push hook optimisation, documentation refresh,
CI improvements, a project website, and CUDA version compatibility testing.

## Success Criteria

- [ ] Pre-push hook parallelized with measurable wall-clock improvement
- [ ] AGENTS.md and developer documentation refreshed
- [ ] Test coverage critical gaps filled; Valgrind CI job added
- [ ] Project website live on GitLab Pages (MkDocs, full gallery, dual feedback buttons)
- [ ] Full test suite verified on CUDA 12.8 and CUDA 13 via parallel CI jobs
- [ ] AWS spot instance workflow polished (error handling, UX, docs)
- [ ] All tests pass

---

## Tasks

### Task 16.1: Optimise Pre-Push Hook

**Estimate:** 2h

Analyze pre-push hook bottlenecks and parallelize where possible. Target: measurable
reduction in wall-clock time for a typical push.

#### Approach

- Profile current hook execution to identify slowest steps
- Parallelize independent checks (compile, lint, test suites) using `#!/bin/bash` and `export -f`
- Remove redundant `sbt compile` (compilation is a subset of `sbt test`)
- Run `scalafix --check` in parallel with `sbt test`
- Run coverage and Valgrind in parallel (GPU vs CPU)
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

Analyze current test coverage to identify critical gaps (under-tested modules or execution
paths), add targeted tests to fill those gaps, and add a Valgrind standalone verification
job to the CI pipeline.

#### Approach

- Run coverage report and identify modules/paths with thin coverage
- Add tests for critical gaps (not chasing an aggregate % target)
- Add Valgrind CI job (separate from the existing compute-sanitizer job)

---

### Task 16.4: Project Website with Feedback Button

**Estimate:** 3h

Create a project website built with **MkDocs**, hosted on **GitLab Pages**
(`https://lilacashes.gitlab.io/menger/`), containing:

- Project overview and feature highlights
- Full gallery of example renders
- Feedback buttons linking to both GitHub issues and GitLab issues

#### Approach

- Set up MkDocs with `mkdocs.yml` and `docs/` source directory
- Write overview page and gallery page (embed existing render images)
- Add GitLab CI `pages` job that runs `mkdocs build` and publishes `site/`
- Add feedback section with direct issue-creation links for both GitHub and GitLab

---

### Task 16.5: Test on CUDA 12 and 13

**Estimate:** 2h

Add a second CI Docker image for CUDA 13 (keeping the existing CUDA 12.8 image as the MR
gate) and run the full test suite on both in parallel.

#### Approach

- Build `optix-cuda:13.x-9.0-<jvm>-<sbt>` Docker image (CUDA 13 + OptiX 9.0)
- Add a parallel `test:cuda13` CI job using that image
- Keep `test:cuda12` (12.8) as the required gate for merge requests
- Document any CUDA-version-specific workarounds discovered

---

### Task 16.6: AWS Spot Instance Workflow Polish

**Estimate:** ~5h (open-ended)

Improve the existing `scripts/nvidia-spot.sh` developer workflow for engineers without a
local GPU. No CLI dispatch feature in scope — focus on quality and usability of the
existing script.

#### Approach

- Audit `nvidia-spot.sh` for error handling gaps and UX rough edges
- Add better error messages, progress feedback, and pre-flight checks
- Review and update related scripts (`backup-spot-state.sh`, `restore-spot-state.sh`, etc.)
- Update `docs/arc42/07-deployment-view.md` and any relevant user-facing docs
- Write or improve a quick-start section in developer documentation

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 16.1 | Optimise pre-push hook | 2h | None |
| 16.2 | Developer docs + AGENTS.md refresh | 2h | None |
| 16.3 | Test coverage + Valgrind CI | 2h | None |
| 16.4 | Project website with feedback button | 3h | None |
| 16.5 | Test on CUDA 12 + 13 | 2h | None |
| 16.6 | AWS spot instance workflow polish | ~5h (open) | None |
| **Total** | | **~16h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing (including CUDA 12.8 + 13)
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] Developer documentation updated
- [ ] Website deployed and accessible at `https://lilacashes.gitlab.io/menger/`
