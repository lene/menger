# AGENTS.md

Guidance for AI coding agents (Claude Code, opencode, etc.) working in this repository.

This is a Scala 3 ray tracer using NVIDIA OptiX (via a C++/CUDA JNI bridge) and LibGDX. Showcase: Menger sponges (3D) and tesseract sponges (4D). Two in-repo modules: `menger-app` (application — CLI, engines, DSL, geometry objects, input, OptiX wrapper) *dependsOn* `menger-geometry` (Menger-specific 4D geometry + caustics; extends `optix-jni` via JNI/CUDA native code; not published). Two domain layers are consumed as separate **published Maven artifacts** maintained in their own repos: `io.github.lene:menger-common` (domain primitives, config) and `io.github.lene:optix-jni` (generic GPU ray tracing, no Menger types).

The user runs **fish shell** on Ubuntu. Most build commands are shell-agnostic; the difference matters only for ad-hoc scripting and env-var syntax.

---

## Critical rules

These are non-negotiable. Violating any of them causes real harm.

1. **Never commit directly to `main`.** Check `git branch --show-current` before any change. If on `main`, switch to (or create) a feature branch first. The active feature branch may live in a worktree under `.worktrees/`.
2. **Never push without explicit user confirmation.** Commit locally, show the diff, wait for "push."
3. **Always monitor the CI pipeline after pushing.** If any failure occur, fix the failures.
3. **Never `git add -A`.** Add files explicitly.
4. **Never commit failing tests.** The pre-push hook enforces this; do not bypass it.
5. **Never rewrite a test to make it pass without investigation.** Failing tests usually catch real bugs. See `docs/TESTING.md`.
6. **Never delete data without explicit user confirmation.** This includes generated artifacts, caches, and reference images.
7. **Never infer values the user should provide** (version numbers, branch names, paths). Ask.
8. **When a skill or instruction says "confirm with user," it is a hard stop.** A prior message in the conversation does not satisfy a fresh checkpoint — ask again.

---

## Definition of Done

**Every task requires both steps — neither alone is sufficient:**

**Step 1: Mark the task done in the sprint doc.** Find the current sprint file via `docs/sprints/SPRINT.md`, open the linked sprint file, check the task's box and update its status. Do this before declaring the task complete.

**Step 2: Run the pre-push hook:**

```
./.git_hooks/pre-push 2>&1 | tee /tmp/pre-push.log
```

It is the single authoritative code-quality gate — unit tests, scalafix, packaging, integration tests (~27 scenarios, ~2,200 unit tests), coverage ratchet (≥80%, max 1% drop), memory leak checks (Valgrind + compute-sanitizer), version consistency. Takes ~8–10 minutes.

Do **not** run individual checks (`sbt test`, `sbt "scalafix --check"`, etc.) as a substitute. The hook is the source of truth and evolves over time.

Always `tee` long-running commands (builds, test runs, render suites) to `/tmp/output.log` so you can re-examine without re-running.

---

## Conventions

### Alpha channel (do not get this wrong)

- `alpha = 0.0` → **fully transparent** (no opacity, no absorption)
- `alpha = 1.0` → **fully opaque** (full opacity, maximum absorption)

Applies everywhere: OptiX shaders, Beer-Lambert absorption, Scala `Color`, all tests.

### Architecture documentation: arc42

Single source of truth: `docs/arc42/README.md`. Consult sections 9 (decisions), 10 (quality), 11 (risks) before architectural changes. Update arc42 if a change affects architecture, quality requirements, or technical debt. Outdated docs are worse than no docs.

### Code style

Scala 3 only. Enforced by Scalafix (`OrganizeImports`, `DisableSyntax` for nulls/returns, no unused imports) and WartRemover (no `var`, `while`, `asInstanceOf`, `throw` in production code). Don't restate these rules in review or commits — trust the tools.

In addition:
- Max line length 100.
- No docstrings; use descriptive names. Comments only for domain-specific reasoning not expressible in code.
- No magic numbers; introduce named constants.
- Tests use `AnyFlatSpec`.
- `@SuppressWarnings` is acceptable in LibGDX integration where the API requires `var`.

### Editing

Manual edits only. Never use `sed` or shell scripts for bulk Scala/C++ refactors — has corrupted code in the past.

### Runtime

Headless rendering needs `__GL_THREADED_OPTIMIZATIONS=0` to avoid crashes in `libnvidia-glcore.so` under `xvfb-run`.

---

## Test failures: investigate first

When a test fails, the default assumption is that the test is catching a real bug. Investigation order:

1. Run the failing test alone, read the actual failure.
2. Check the test's git history and recent changes to the code under test.
3. Decide which is wrong: implementation, test expectation, or both.
4. Document the investigation in the commit message.

Full protocol with decision tree, example commit message, and red flags: `docs/TESTING.md`.

**Red flags that require asking the user:** multiple tests failing after a "simple" refactor, visual tests fail but unit tests pass, only some similar tests fail, a long-stable test starts failing.

---

## Rendering changes

If a commit changes anything that can affect rendered pixel output, the integration suite (`scripts/integration-tests.sh`) must run on the same commit, and any reference-image diffs must be resolved in that same commit (or an immediately-following test-only commit on the same branch).

**Do not let the pre-push hook be the discovery mechanism.** Stale references accumulating across multiple commits make bisection painful later.

What counts as a rendering change, the sequential-vs-parallel-mode trap, and the full discipline: `docs/RENDERING.md`.

Quick rule for local verification of rendering changes: run sequential mode to avoid GPU-contention flakes:
```
PARALLEL_MODE=false ./scripts/integration-tests.sh ./menger-app-$VERSION/bin/menger-app
```

Every new rendering feature (material preset, object type, shader path, CLI parameter) must be added to **both** `scripts/integration-tests.sh` (headless regression) and `scripts/manual-test.sh` (human visual verification). Verify both scripts have at least one test exercising the new code before considering the feature done.

---

## Common commands

```
sbt compile                          # All modules (includes C++/CUDA)
sbt test                             # All tests
sbt "testOnly ClassName"             # Specific Scala test
sbt run                              # Run application
sbt "scalafix --check"               # Code quality check
sbt "project optixJni" nativeCompile # C++/CUDA only
sbt "project optixJni" nativeTest    # C++ Google Test suite
rm -rf optix-jni/target/native ; sbt "project optixJni" compile  # Clean rebuild of native
```

Pipeline monitoring after push:
```
glab ci view       # Watch latest pipeline
glab ci status     # Current status
```

Detailed troubleshooting (CUDA error 718, OptiX SDK/driver matching, PTX-not-found, Docker permissions): `docs/TROUBLESHOOTING.md`.

---

## Release workflow

Use the `/release-checklist` skill. Version must be updated in four files: `menger-app/build.sbt`, `.gitlab-ci.yml` (DEPLOYABLE_VERSION), `menger-app/src/main/scala/menger/MengerCLIOptions.scala`, `docs/guide/user-guide.md`. The pre-push hook validates consistency across all four.

---

## Pointers

| Where | What |
|---|---|
| `docs/arc42/README.md` | Architecture (authoritative) |
| `docs/ENFORCEMENT.md` | Policy → mechanism map; open enforcement gaps |
| `docs/TESTING.md` | Test failure protocol, investigation procedure |
| `docs/RENDERING.md` | Rendering-change discipline, integration suite |
| `docs/TROUBLESHOOTING.md` | Common environment/build issues |
| `docs/sprints/SPRINT.md` | Current sprint pointer |
| `CHANGELOG.md` | Version history (keepachangelog format) |
| `CODE_IMPROVEMENTS.md` | Open code-quality findings (resolved items deleted, not archived) |
| `optix-jni/README.md` | OptiX JNI module details |

---

## Agent tool notes

### Bash tool — **needs verification, do not trust until confirmed**

These workarounds were documented for an older agent setup and may no longer apply. Verify on first use; remove from this file if obsolete.

- `pkexec` instead of `sudo` — sudo prompts for a password on a TTY the agent doesn't have.
- Command chaining with `;` instead of `&&` — was reported to fail. Test with `echo a && echo b` before relying on this rule.
- `$()` command substitution — was reported to fail. Test before relying on this rule.

### Output style for routine tasks

For straightforward edits, status reports, and short answers, prefer a dense style:
- One atomic fact per line. Abbreviations OK (fn, cfg, impl, deps, req, res, ctx, err, ret).
- Diff lines only (`+`/`-`/`~`); never repeat unchanged code.
- Symbols welcome: → (causes), + (adds), − (removes), ~ (modifies), ∴ (therefore).
- No narration, no hedging.

This style is **not appropriate** for code review, architecture discussion, test-failure investigation, design tradeoff analysis, or anywhere a clarifying question would be more useful than a terse answer. For those, write prose at whatever length the situation needs.

---

## Miscellaneous

- Capture screenshots with `scrot` if needed.
- When fetching from git, use `--all --tags`.

---

## A note on this file's length

This file is intentionally short. Agents reliably internalize ~200-line instruction files; beyond that, attention drops and rules are selectively ignored. If you (a future maintainer or agent) feel the urge to add a section, ask first whether the content is *policy* (belongs here) or *reference* (belongs in a linked doc, skill, or troubleshooting file).
