# Sprint 28: Industrial-Strength Release & QA

**Sprint:** 28 - Industrial-Strength Release & QA
**Status:** Not Started
**Estimate:** ~26 hours
**Branch:** `feature/sprint-28`
**Dependencies:** None. Builds on Sprint 27.0's CI hardening of the standalone
`menger-common` and `optix-jni` repositories (MiMa, coverage ratchet, release gates),
which covered the *libraries*; this sprint covers the *app repo* and the cross-repo
release process.
**Feature ID:** F14 in [FEATURE_DEPENDENCIES.md](FEATURE_DEPENDENCIES.md)

---

## Goal

Make releases boring and regressions loud. One-command release preparation, sanitizer
and performance gates that run on a schedule instead of only at the developer's
discretion, supply-chain artifacts (SBOM), and a documented flaky-test policy. This
sprint front-loads process quality so every following feature sprint ships through
hardened machinery.

---

## Success Criteria

- [ ] `scripts/release.sh --dry-run <version>` performs every mechanical step of
      `/release-checklist` (4-file version bump, consistency check, changelog scaffold)
      without human copy-paste
- [ ] Nightly scheduled pipeline runs full integration + sanitizers + performance
      benchmarks without manual triggering
- [ ] A >15 % performance regression on the benchmark scene set fails the nightly job
- [ ] SBOM (CycloneDX) generated for menger-app JVM dependencies and attached to
      release artifacts
- [ ] Dependency-update automation replaces the broken sbt-updates plugin
- [ ] Flaky-test policy documented in `docs/TESTING.md` (covers the known
      sponge-volume / tesseract-with-material integration flakes)
- [ ] All tests pass

---

## Tasks

### Task 28.1: Release Automation Script

**Estimate:** 4h

`scripts/release.sh` automating the mechanical steps of the `/release-checklist` skill.

**Implementation:**
- Bump version in all four files: `menger-app/build.sbt`, `.gitlab-ci.yml`
  (DEPLOYABLE_VERSION), `MengerCLIOptions.scala`, `docs/guide/user-guide.md`
- Reuse the pre-push hook's version-consistency check as the validation step
- Scaffold a CHANGELOG.md section from `git log <last-tag>..HEAD --oneline`
  (conventional-commit prefixes → keepachangelog categories)
- `--dry-run` prints the plan without writing; default refuses to run on a dirty tree
- The script prepares; it never pushes or tags on its own — tag creation stays a
  manual, confirmed step per repository policy

---

### Task 28.2: Dependency-Update Automation

**Estimate:** 3h

sbt-updates is incompatible with sbt 1.12.6 (NPE on load; see TODO.md). Replace it.

**Implementation:**
- Evaluate: Scala Steward (self-hosted GitLab job) vs. Renovate (GitLab app) — spike
  both against this repo, pick one, document the decision in arc42 §9
- Weekly scheduled CI job opening MRs for dependency bumps
- Native deps (OptiX SDK, CUDA, libav) are excluded — driver/SDK matching is manual by
  necessity (see `docs/TROUBLESHOOTING.md`); document the exclusion

---

### Task 28.3: Sanitizer Jobs in CI

**Estimate:** 5h

The pre-push hook runs Valgrind and compute-sanitizer locally; CI does not. Promote them.

**Implementation:**
- ASan + UBSan build profile for `menger-geometry` native code
  (`-fsanitize=address,undefined`, CMake build type `Sanitize`); run the C++ Google
  Test suite under it in a scheduled CI job
- compute-sanitizer (memcheck) job on the GPU runner against the OptiX integration
  smoke subset — full suite under compute-sanitizer is too slow for nightly; pick the
  3–4 scenarios with the most allocation churn
- Suppression files checked in and documented (known CUDA runtime leaks)

---

### Task 28.4: Performance-Regression Tracking

**Estimate:** 6h

**Implementation:**
- Benchmark scene set (fixed seeds, fixed resolution): sphere baseline,
  sponge-volume level 4, menger4d level 3, IBL + accumulation scene
- Emit `--stats` metrics (ms/frame, ms/Mray, rays by type) as JSON
  (`--stats-json <file>` CLI addition — small Scala change)
- Nightly job renders the set, compares against a rolling baseline stored as a CI
  artifact (median of last 5 green runs); >15 % ms/frame regression fails the job
- GPU-runner variance is the main risk: pin the runner, use medians of 3 runs,
  and gate on relative—not absolute—numbers

---

### Task 28.5: SBOM and Supply-Chain Artifacts

**Estimate:** 3h

**Implementation:**
- sbt CycloneDX plugin for JVM dependency SBOM; attach to release artifacts in CI
- Document native third-party components (OptiX SDK, CUDA, libav with LGPL shared
  linking note, stb_image etc.) in a hand-maintained `NOTICE` section referenced from
  the SBOM — native deps aren't captured by the JVM tool
- Maven Central artifacts are already signed (Sonatype requirement); document the
  signing chain in arc42 §7

---

### Task 28.6: Nightly Pipeline Schedule + Flaky-Test Policy

**Estimate:** 2h

**Implementation:**
- GitLab scheduled pipeline (nightly): full integration suite, sanitizer jobs (28.3),
  performance benchmarks (28.4)
- Flaky-test policy in `docs/TESTING.md`: known flaky scenarios (sponge-volume,
  tesseract-with-material image comparisons) get one automatic retry in CI; a test may
  stay on the retry list max 2 sprints before it must be fixed or its flakiness
  root-caused; the list lives in the integration script, not in skipped tests

---

### Task 28.7: Documentation

**Estimate:** 3h

- arc42 §10 (quality requirements): add measurable release/QA quality scenarios
- arc42 §11 (risks): update for perf-variance and supply-chain risks
- Update the `/release-checklist` skill to call `scripts/release.sh`
- CHANGELOG.md entry

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 28.1 | Release automation script | 4h |
| 28.2 | Dependency-update automation | 3h |
| 28.3 | Sanitizer jobs in CI | 5h |
| 28.4 | Performance-regression tracking | 6h |
| 28.5 | SBOM + supply-chain artifacts | 3h |
| 28.6 | Nightly schedule + flaky-test policy | 2h |
| 28.7 | Documentation | 3h |
| **Total** | | **~26h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
- [ ] At least one nightly pipeline has run green end-to-end before sprint close
