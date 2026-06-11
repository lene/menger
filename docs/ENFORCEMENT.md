# Enforcement Audit: Policy → Mechanism Map

Living document. Reviewed at sprint close. Every ❌ row has an open issue;
resolve by implementing a gate or consciously accepting the gap.

**Status legend:**
- ✅ Enforced — structural gate; violations are rejected automatically
- ⚠️ Partial — enforced with known gaps
- 🤖 AI policy — behavioural instruction; mechanically unenforceable by design
- ❌ Unenforced — open issue exists; see linked issue

---

## Commit hygiene

| Policy | Source | Mechanism | Status |
|--------|--------|-----------|--------|
| No commits directly on `main` | AGENTS.md §Critical rules | `standards/hooks/check-branch.sh` (pre-commit) | ✅ |
| No conflict markers or whitespace errors in staged diff | AGENTS.md §Editing | `standards/hooks/check-staged-hygiene.sh` (pre-commit) | ✅ |
| No `println()` in production Scala sources | AGENTS.md §Code style | `standards/hooks/check-staged-hygiene.sh` (pre-commit) | ✅ |
| No files > 5 MB staged | AGENTS.md §Editing | `standards/hooks/check-staged-hygiene.sh` (pre-commit) | ✅ |
| Never `git add -A` (no accidental sensitive-file staging) | AGENTS.md §Critical rules | 🤖 AI policy | 🤖 |
| Never push without explicit user confirmation | AGENTS.md §Critical rules | 🤖 AI policy | 🤖 |

---

## Test discipline

| Policy | Source | Mechanism | Status |
|--------|--------|-----------|--------|
| Tests must pass before commit | AGENTS.md §Critical rules | pre-commit: `sbt test` (Scala/native changes) | ✅ |
| Tests must pass before push | AGENTS.md §Critical rules | pre-push Phase 3: `sbt test` | ✅ |
| Modified/deleted test files require `Test-Change:` trailer | AGENTS.md §Test failures | `standards/hooks/check-test-justification.sh` (pre-push Phase 0) | ✅ |
| Test failure investigation protocol documented | AGENTS.md §Test failures | `docs/TESTING.md` | ✅ |
| Coverage ≥ 80 %, max 1 % drop | AGENTS.md §Definition of Done | pre-push Phase 4: coverage ratchet | ✅ |

---

## Rendering discipline

| Policy | Source | Mechanism | Status |
|--------|--------|-----------|--------|
| Rendering-relevant commits must update reference images or carry `No-Render-Impact:` | AGENTS.md §Rendering changes | `standards/hooks/check-rendering-discipline.sh` (pre-push Phase 0) | ✅ |
| Reference-image diffs resolved in same commit (not discovered at pre-push) | AGENTS.md §Rendering changes | Same hook: push-range check | ✅ |
| New rendering feature added to `integration-tests.sh` | AGENTS.md §Rendering changes | Unenforced | ❌ [#2] |
| New rendering feature added to `scripts/manual-test.sh` | AGENTS.md §Rendering changes | Unenforced | ❌ [#3] |
| Alpha: 0.0 = transparent, 1.0 = opaque (never inverted) | AGENTS.md §Conventions | Unenforced | ❌ [#4] |
| Rendering discipline fully documented | AGENTS.md §Rendering changes | `docs/RENDERING.md` | ✅ |

---

## Code quality — Scala

| Policy | Source | Mechanism | Status |
|--------|--------|-----------|--------|
| Scalafix passes (`OrganizeImports`, `DisableSyntax`) | AGENTS.md §Code style | pre-commit + pre-push Phase 3: `sbt scalafix --check` | ✅ |
| WartRemover: no `var`, `while`, `asInstanceOf`, `throw` in production | AGENTS.md §Code style | WartRemover via `sbt compile` (pre-commit + pre-push) | ✅ |
| No unused imports | AGENTS.md §Code style | `OrganizeImports` scalafix rule | ✅ |
| No null references | AGENTS.md §Code style | `DisableSyntax` scalafix rule | ✅ |
| arc42 updated on architectural changes | AGENTS.md §Architecture | Unenforced | ❌ [#6] |

---

## Code quality — C++/CUDA

| Policy | Source | Mechanism | Status |
|--------|--------|-----------|--------|
| cppcheck (warnings, performance, portability) | AGENTS.md §Code style | pre-push Phase 4: `run_cppcheck` (native changes) | ✅ |
| clang-tidy | AGENTS.md §Code style | pre-push Phase 4: `run_clang_tidy` (native changes) | ✅ |
| CUDA environment present before native build | AGENTS.md §Runtime | pre-commit + pre-push Phase 1: env validation | ✅ |

---

## Version & release

| Policy | Source | Mechanism | Status |
|--------|--------|-----------|--------|
| Version consistent across `build.sbt`, `.gitlab-ci.yml`, `MengerCLIOptions.scala`, `user-guide.md` | AGENTS.md §Release workflow | pre-push Phase 1: `scripts/check-version-consistency.sh` | ✅ |
| Git tag not already used | AGENTS.md §Release workflow | pre-push Phase 1: tag availability check | ✅ |
| GitLab CI YAML valid | AGENTS.md §Release workflow | pre-push Phase 1: CI lint via API (`.gitlab-ci.yml` changes) | ✅ |
| Release triggers on MR merge (unless `NORELEASE` label) | Sprint 28.5 | GitLab CI `CreateRelease` job on main-branch pipeline | ✅ |
| Released package installs and renders on user's OS | Sprint 28.5 | `InstallProof` CI job on tag pipeline | ✅ |
| CHANGELOG.md updated at release | AGENTS.md §Release workflow | CI `ChangelogIsUpdated` job: checks top entry matches current tag and date | ✅ |

---

## Cross-repo standards

| Policy | Source | Mechanism | Status |
|--------|--------|-----------|--------|
| Shared configs byte-identical across menger / menger-common / optix-jni | Sprint 28.1 | Scheduled CI job: `scripts/check-standards-drift.sh` (daily) | ✅ |
| Local standards parity before every commit | Sprint 28.2 | pre-commit: `scripts/check-standards-drift.sh --local` | ✅ |
| Standards sync to sibling repos reviewed like any code change | Sprint 28.1 | Manual: `scripts/sync-standards.sh` (no silent cross-repo writes) | ⚠️ |

---

## MR / merge gate

| Policy | Source | Mechanism | Status |
|--------|--------|-----------|--------|
| Every MR reviewed by ≥ 2 AI model families | Sprint 28.4 | GitLab CI `AIReview` job posts structured comments | ✅ |
| Pipeline must succeed before merge | Sprint 28.6 | GitLab project setting: *Pipelines must succeed* | ✅ |
| All discussions must be resolved before merge | Sprint 28.6 | GitLab project setting: *All discussions must be resolved* | ✅ |
| Always monitor CI pipeline after push | AGENTS.md §Critical rules | 🤖 AI policy | 🤖 |

---

## Performance

| Policy | Source | Mechanism | Status |
|--------|--------|-----------|--------|
| No silent performance regressions (> 15 % ms/frame) | Sprint 28.7 | CI `PerfCheck` job (`allow_failure: true`): `benchmark.sh` compares 4 scenes against `perf-baseline.json` (15 % threshold) | ⚠️ advisory |

---

## Agentic workflow (AI-only policies)

These policies govern AI agent behaviour and are structurally unenforceable by
a gate. They are listed here for completeness and to confirm the gap is
consciously accepted.

| Policy | Source | Status |
|--------|--------|--------|
| Never infer values the user should provide (version numbers, branch names, paths) | AGENTS.md §Critical rules | 🤖 |
| Never delete data without explicit user confirmation | AGENTS.md §Critical rules | 🤖 |
| When a skill says "confirm with user", it is a hard stop | AGENTS.md §Critical rules | 🤖 |

---

## Open issues

| # | Policy gap | Action |
|---|-----------|--------|
| 2 | No check that a new rendering feature is added to `integration-tests.sh` | Implement hook or CI check — [#156](https://gitlab.com/lilacashes/menger/-/work_items/156) |
| 3 | No check that a new rendering feature is added to `manual-test.sh` | Implement hook or CI check — [#157](https://gitlab.com/lilacashes/menger/-/work_items/157) |
| 4 | Alpha-channel convention (0.0 = transparent) has no static check | Add WartRemover rule or comment-linter — [#158](https://gitlab.com/lilacashes/menger/-/work_items/158) |
| 6 | No check that arc42 is updated when architecture-relevant files change | Implement hook using `standards/architecture-paths.txt` — [#160](https://gitlab.com/lilacashes/menger/-/work_items/160) |
| 7 | CHANGELOG.md update not enforced at release time | Add pre-push or CI check for CHANGELOG date/entry — [#161](https://gitlab.com/lilacashes/menger/-/work_items/161) |
| 8 | Performance regression guard not yet implemented | Task 28.7 — [#162](https://gitlab.com/lilacashes/menger/-/work_items/162) |
