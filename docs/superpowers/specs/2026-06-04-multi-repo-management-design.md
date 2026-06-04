# Design: Multi-Repo Management & CI Strategy

**Date:** 2026-06-04  
**Repos:** menger-app (GitLab primary, GitHub mirror), menger-common (GitHub), optix-jni (GitHub)

---

## Context

Sprint 26 split menger-common and optix-jni out as independently published artifacts. Three repos now need to be developed coherently while remaining independently deployable. This document defines the local workspace layout, agent instruction strategy, CI stage conventions, version coordination, and open implementation tasks.

---

## 1. Local Repo Layout

```
~/workspace/menger-toplevel/     ← private git repo (new)
  menger-shared.md               ← shared context for multi-repo AI sessions
  .gitignore                     ← ignores menger/, menger-common/, optix-jni/
  menger/                        ← GitLab primary / GitHub mirror (gitignored)
  menger-common/                 ← GitHub primary (gitignored)
  optix-jni/                     ← GitHub primary (gitignored)
```

`menger-toplevel` tracks only shared files. The three sub-repos are independent git repos co-located under it. Starting Claude Code from `menger-toplevel/` gives the agent visibility into all three via relative paths without any git nesting.

---

## 2. Agent Instruction Strategy (CLAUDE.md)

Each repo's `CLAUDE.md` is **self-contained** — all critical rules present, no @-include dependencies. A standalone clone works correctly for any contributor.

Each repo's `CLAUDE.md` includes this note:

> For multi-repo sessions (cross-repo changes, version bumps, shared sprint work), start Claude Code from `~/workspace/menger-toplevel/`. The `menger-shared.md` file there provides cross-repo workflow context.

### `menger-toplevel/menger-shared.md` contains
- Cross-repo version coordination workflow
- Repo locations and primary remotes
- Sprint/roadmap pointer
- Multi-repo session workflow for AI agents

### Critical rules in every repo's own `CLAUDE.md` (never delegated)
- Never commit to main
- Never push without explicit user confirmation
- Never `git add -A`
- Never commit failing tests
- Alpha channel convention: `0.0` = fully transparent, `1.0` = fully opaque
- Pre-push hook is the DoD gate

---

## 3. CI Stage Map

Five logical stages across all three repos. Job content adapts to each project's nature; stage names match.

| Stage | menger-app (GitLab) | menger-common (GitHub Actions) | optix-jni (GitHub Actions) |
|-------|--------------------|-----------------------------|--------------------------|
| **preconditions** | version consistency, changelog updated, tag format | version consistency, changelog updated | version consistency, changelog updated |
| **check** | scalafix, cppcheck, clang-tidy | scalafix, ArchUnit | scalafix, ArchUnit, cppcheck, clang-tidy, doc completeness |
| **test** | unit tests + coverage + ArchUnit | unit tests + coverage | Google Test (C++) + no-GPU Scala tests + coverage |
| **build** | deployable tarball | JAR | native .so + JAR |
| **test-built** | integration tests (`nvidia` runner) | *(none — no runtime artifact)* | OptiX smoke + Java smoke + Kotlin smoke (`nvidia` self-hosted) |
| **publish** | GitLab release + push to GitHub | GitHub release + Maven Central | GitHub release + Maven Central |

### Runner label: `nvidia`
- menger-app: existing `nvidia`-tagged GitLab runner
- menger-common + optix-jni: self-hosted GitHub Actions runner on same machine, registered with label `nvidia`; GPU jobs use `runs-on: [self-hosted, nvidia]`

### Doc completeness check (optix-jni, `check` stage)
Bidirectional check:
- **Forward**: every public function in the JNI API (Scala bindings) has a Scaladoc comment
- **Backward**: every documented function exists in the API (catches stale docs after removals)

Implementation: script diffing the public method list from Scala JNI bindings against Scaladoc output. Needs separate design + implementation task.

### ArchUnit scope
- menger-common: package naming rules, no circular dependencies
- optix-jni: package naming for Scala JNI bindings layer

### Changelog
All three repos track changes in keepachangelog format. `preconditions` stage enforces that the changelog is updated on every MR/PR (same as menger-app's existing `ChangelogIsUpdated` job).

---

## 4. Version Coordination

Manual. No automation.

1. Implement and test change in menger-common or optix-jni
2. Merge to main, tag release, publish to Maven Central via CI
3. In menger-app, bump the version in `build.sbt` explicitly
4. Run pre-push hook, push

No bot PRs, no SNAPSHOT versions in production.

---

## 5. Open Implementation Tasks

| # | Task | Repo | Notes |
|---|------|------|-------|
| 1 | Create `menger-toplevel` repo | new | Init, .gitignore, menger-shared.md, clone sub-repos |
| 2 | Register self-hosted GitHub Actions runner | infra | Label `nvidia`; connect to menger-common + optix-jni repos |
| 3 | GitHub Actions CI for menger-common | menger-common | Implement stage map; changelog + ArchUnit |
| 4 | GitHub Actions CI for optix-jni | optix-jni | Implement stage map; Java + Kotlin smoke test projects |
| 5 | Doc completeness check script | optix-jni | Bidirectional JNI API ↔ Scaladoc; design first |
| 6 | Add changelogs | menger-common, optix-jni | keepachangelog format; preconditions job enforces |
| 7 | Add ArchUnit tests | menger-common, optix-jni | Package naming + no circular deps |
