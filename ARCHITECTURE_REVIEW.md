# Architecture Review — Sprint 31 Close

**Date:** 2026-06-29
**Reviewed by:** Hermes Agent (code-review + arch-review skills)
**Delta:** 20 commits, 48 files (+2704/-342)

---

## Axis Scorecard

| Axis | Arc42 § | Grade | Key Gap |
|------|---------|-------|---------|
| Soundness | §5 | ✅ Guarded | Module deps enforced by ArchUnit; L-system code placed in correct packages |
| Maturity | §10/§11 | ⚠️ Partial | Tests pass but no render-determinism gate; JNI leak gate is CI-only |
| Evolvability | §9 | ⚠️ Partial | Adding a new geometry type still touches 6+ files; dispatch partially unified |
| Performance | §10.4 | ⚠️ Partial | PerfCheck exists but baselines not committed; no regression guard |

---

## Fitness-Function Status Table

| Function | Status | Enforced? |
|----------|--------|:---:|
| Module dependency direction (ArchUnit) | ✅ Active | ✅ CI + pre-push |
| Package placement (ArchitecturePhase2Spec) | ✅ Active | ✅ CI + pre-push |
| Coverage ratchet (≥80%, max 1% drop) | ✅ Active | ✅ CI + pre-push |
| Version consistency (4 files) | ✅ Active | ✅ CI + pre-push |
| Changelog updated | ✅ Active | ✅ CI |
| JNI native leak gate (Valgrind + compute-sanitizer) | ✅ Active | ✅ CI (allow_failure) |
| CI YAML lint | ✅ Active | ✅ pre-push |
| Render determinism | ❌ Missing | ❌ None |
| Performance budget (P1/P2) | ⚠️ Advisory | ⚠️ CI (allow_failure) |
| Script parity (integration ⊇ manual) | ❌ Missing | ❌ None |
| Object-type dispatch completeness | ⚠️ Test only | ⚠️ Unit test (not blocking) |

---

## Findings

### A1 — Object-type dispatch still triplicated (Evolvability, High)

**Location:** `GeometryRegistry.scala:28-52`, `InteractiveEngine.scala:556-600`,
`RenderModeSelector.scala:18-35`
**Evidence:** The 2026-06-12 patch synced `sierpinski4d`/`hexadecachoron4d` across three
if/else chains, but the root cause (three hand-maintained dispatch tables) remains.
Adding a new type still touches 6-7 files.
**Enforcement:** Partially guarded — the completeness unit test catches missing entries, but
only after the code is written. No compile-time guard that a new `ObjectType.VALID_TYPES`
entry must have a corresponding builder.
**Fitness function:** Single source-of-truth registry table consumed by all three sites.
**See:** ARCHITECTURE_BACKLOG.md T1

### A2 — Performance is ungoverned (Performance Architecture, High)

**Location:** `scripts/benchmark.sh`, `.gitlab-ci.yml:PerfCheck`
**Evidence:** PerfCheck exists as a CI job with `allow_failure: true`. arc42 §10.4 budgets
(P1 <5s, P2 <500ms) are declared but no baselines are committed in `perf-baseline.json`.
PerfCheck runs but its results don't block anything.
**Enforcement:** Unguarded — a 10× render-time regression would not fail CI.
**Fitness function:** PerfCheck as a blocking job with committed baselines.
**See:** ARCHITECTURE_BACKLOG.md T2

### A3 — Native memory-leak gate is CI-only (Maturity, Medium)

**Location:** `.git_hooks/pre-push:343-375`, `.gitlab-ci.yml:NativeLeakCheck`
**Evidence:** The pre-push hook's Valgrind/compute-sanitizer stages are conditional on
`HAS_NATIVE` and skip gracefully when tools are not installed. In CI, `NativeLeakCheck`
runs as an advisory job (`allow_failure: true`). Native leaks introduced in a PR are
caught only when CI runs — not during local development.
**Enforcement:** Partially guarded — CI covers it but pre-push may silently skip.
**Fitness function:** Pre-push Valgrind gate should fail loudly when native files change
and tools are missing (install instructions + exit 1 instead of return 0).
**See:** ARCHITECTURE_BACKLOG.md T3

### A4 — String-based builder dispatch in LSystemSceneBuilder (Soundness, Low)

**Location:** `LSystemSceneBuilder.scala:51-55`
**Evidence:** `resolveSubBuilder` matches on `"curve"`, `"sphere"`, `"cone"` strings
with a wildcard default to `CurveSceneBuilder`. A typo in `emitRun`'s ObjectSpec
construction would silently route to the wrong builder.
**Enforcement:** Unguarded — no compile-time check that the objectType strings emitted
by the turtle match the dispatch table.
**Fitness function:** Enum or sealed trait for sub-builder type, consumed by pattern match.

### A5 — 4D presets inconsistently placed (Evolvability, Low)

**Location:** `LSystemTurtle4D.scala:253-264`, `LSystemPresets.scala`
**Evidence:** 3D presets live in `menger.objects.LSystemPresets` (shared by CLI and DSL
paths). 4D presets (`HilbertCurve4D`, `Tree4D`) are hardcoded in the `LSystemTurtle4D`
companion object, instantiated with full grammar strings — not exposed to the DSL or CLI
preset lookup. Adding a new 4D preset requires touching a different file than adding a
3D preset.
**Enforcement:** Unguarded — convention only.
**Fitness function:** Move 4D presets into `LSystemPresets` (or a 4D variant) with the
same interface.

---

## arc42 Coherence

| Section | Status | Action |
|---------|--------|--------|
| §5 Building Block View | ✅ Accurate | No changes needed — module structure unchanged |
| §9 Architectural Decisions | ⚠️ Stale | Missing: OptiX-as-sole-backend ADR (T11), L-system design decisions |
| §10 Quality Requirements | ⚠️ Stale | P1/P2 marked "Validated" but never measured; caustics ladder claims 8 rungs, 0 exist |
| §11 Risks | ✅ Accurate | JNI leak risk tracked; CI gate present |

---

## Summary (updated Sprint 32 close)

The architecture is fundamentally sound — module boundaries are enforced, test coverage
is high, and the L-system integration follows existing patterns (SceneBuilder, ObjectSpec).

**Sprint 32 closure:** T1, T2, T5, T7, T9, T10, T11, A4, A5 all completed. The dispatch
drift (F1) is eliminated via TypeRegistry. Performance budgets (T2) are governed by
PerfCheck with committed baselines. Native-binding discipline (T5) is enforced by
ArchUnit. Script parity (T9), fault injection (T7), and fast-path guard (T10) add
automated fitness functions. Two remaining items defer to Sprint 33: T3 (native leak
gate) and T6 (caustics C1-C4).
