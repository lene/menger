# Architecture Backlog

Schedulable tasks derived from `ARCHITECTURE_REVIEW.md` (2026-06-12, first `/arch-review` run on
`feature/sprint-27`). One task per problematic finding, ordered by priority, with effort estimates
so they can be slotted into sprints. Each task names the **fitness function** it adds — the goal is
to convert convention-maintained invariants into automated guards.

Effort key: **XS** < 1h · **S** ≈ half-day · **M** ≈ 1–2 d · **L** ≈ 2–4 d.

Already actioned outside this backlog:
- **F5** (CLAUDE.md module-map drift) — fixed (2026-06-12).
- **F1 symptom** (4D dispatch gap) — `sierpinski4d`/`hexadecachoron4d` wired into `GeometryRegistry`
  + `RenderModeSelector`, completeness test added, integration coverage added (2026-06-12). The
  *root-cause* unification remains below as **T1**.
- **F4** (opaque types across the JNI seam) — done in **Sprint 27.12** (`InstanceId.scala`,
  `-1` allocation failures fail builds at the boundary). Removed from the backlog.
- **F9** (DSL/scene-config trust-boundary validation) — done in **Sprint 27.11**
  (`M-sceneb-validate-bypass` closed; scene builders validate before renderer allocation).
  Removed from the backlog.

---

## Summary table

| # | Task | Finding | Effort | Priority | Status |
|---|------|---------|--------|----------|--------|
| T1 | Unify object-type dispatch into one registry table | F1 | M (1.5–2 d) | High | ✅ Sprint 32 |
| T2 | Performance governance (PerfCheck + benchmark baselines) | F2 | M (1 d) | High | ✅ Sprint 32 |
| T3 | In-repo native memory-leak gate | F3 | M (1 d) | High | 🔜 |
| T9 | Script-parity fitness function | F1 | S (2–4 h) | Med-High | ✅ Sprint 32 |
| T5 | Module-scoped ArchUnit native rule + rename `menger.geometry` | F6 | M (4–8 h) | Medium | ✅ Sprint 32 |
| T6 | Caustics ladder analytic rungs C1–C4 | F7 | L (2–3 d) | Medium | 🔜 Sprint 33 |
| T7 | Determinism + JNI fault-injection test kinds | F8 | S (~1 d) | Medium | ✅ Sprint 32 |
| T10 | Fast-path regression guard | Pos.1 | S (3–5 h) | Medium | ✅ Sprint 32 |
| T11 | Record OptiX-as-sole-backend as a deliberate ADR | F-backend | XS (~30 m) | Low | ✅ Sprint 32 |

Total if all scheduled: **~7–10 dev-days** (T4/T8 completed in Sprint 27 and removed).

---

## Tasks

### T1 — Unify object-type dispatch (High, M 1.5–2 d) — Finding 1

**Problem:** `ObjectType` dispatch is triplicated across `GeometryRegistry.builderFor`,
`RenderModeSelector`, and `InteractiveEngine.buildScene4DTrackedOrFallback`. The 2026-06-12 patch
re-synced them for `sierpinski4d`/`hexadecachoron4d`, but the structural risk (three hand-maintained
if/else chains that can drift again) remains.
**Do:** Introduce a single source of truth — a registry list/`Map` of
`(predicate, builderFactory)` — consumed by all three sites. `InteractiveEngine` still supplies its
recorder callbacks, but the *type→builder* decision lives in one place.
**Fitness function:** the completeness test from the 2026-06-12 patch (every `ObjectType.VALID_TYPES`
entry resolves to exactly one builder) becomes the guard over the unified table.
**Done when:** the three if/else chains are gone; adding a type touches one table + one builder.

### T2 — Performance governance (High, M 1 d) — Finding 2

**Problem:** No executing performance fitness function on this branch; arc42 §10.4 budgets (P1 <5s,
P2 <500ms) are marked "Validated" but never re-measured. `benchmark.sh`/`PerfCheck` reportedly exist
in the sprint-28 worktree but are not present here.
**Do:** Confirm sprint-28's `benchmark.sh` + `PerfCheck` land on the integration branch; wire them to
the §10.4 budgets; emit machine-readable results + a committed baseline; run `PerfCheck` as
`allow_failure: true`, then promote P1/P2 to blocking once baselines are stable.
**Fitness function:** `PerfCheck` CI job asserting P1/P2 against the committed baseline.
**Done when:** a multiple-× regression in subdivision or render time fails (or at least flags) CI.

### T3 — In-repo native memory-leak gate (High, M 1 d) — Finding 3

**Problem:** The Valgrind/compute-sanitizer steps print "skipped — standalone repository"; the
shipping in-repo native code (`menger-geometry/src/main/native/`, `libmengergeometry.so`) has no leak
gate, yet arc42 §11.4 lists JNI leaks as a *mitigated* risk.
**Do:** Add a CI job running `compute-sanitizer --leak-check` (+ valgrind where feasible) over
`menger-geometry` `nativeTest` on the CUDA runner, OR restore the pre-push body gated on
`HAS_NATIVE` changes. Risk: CUDA-runner configuration.
**Fitness function:** native leak check over the in-repo module in CI.
**Done when:** a leak introduced in `MengerJNIBindings.cpp` is caught automatically.

### T9 — Script-parity fitness function (Med-High, S 2–4 h) — Finding 1

**Problem:** CLAUDE.md mandates every rendering feature appear in both `integration-tests.sh` and
`manual-test.sh`; this drifted (the 2026-06-12 patch fixed the current gap, but nothing prevents
recurrence).
**Do:** A test that greps both scripts for `type=<...>` tokens and asserts
`integration ⊇ manual`-equivalent coverage over `ObjectType.VALID_TYPES`; fail CI on divergence.
**Fitness function:** the parity test itself.
**Done when:** adding a type to only one script fails CI.

### T5 — Module-scoped ArchUnit native rule + rename `menger.geometry` (Medium, M 4–8 h) — Finding 6

**Problem:** Two ArchUnit rules use disagreeing scopes (package whitelist `io.github.lene.optix..`
vs path filter `.*/optix-jni/.*`); `MengerRenderer`'s native overrides in `menger-geometry` escape
the optix-jni-scoped rule. Separately, package `menger.geometry` (named for pure geometry) contains
only an FFmpeg video JNI loader.
**Do:** Make the native-binding rule module-path-based (bindings originate only from expected module
locations; `menger-geometry` classes *extend* the published optix-jni surface, never duplicate it).
Rename `menger.geometry` → `menger.video`, or assert it has no `@native` methods.
**Fitness function:** module-path-scoped native-binding ArchUnit rule.
**Done when:** a stray `@native` in the wrong module fails the architecture test.

### T6 — Caustics ladder analytic rungs C1–C4 (Medium, L 2–3 d) — Finding 7

**Problem:** The C1–C8 caustics ladder in arc42 §10 / `docs/caustics/CAUSTICS_TEST_LADDER.md` is
0/8 implemented; no SSIM is computed anywhere despite C8's "SSIM > 0.90" acceptance gate. The only
real safety net is two pixel-diff checks.
**Do:** Implement the analytic rungs C1–C4 as `AnyFlatSpec` determinism tests (no GPU needed — they
catch refraction-math regressions); mark C5–C8 "not implemented" in arc42 §10 so the doc stops
over-claiming; defer the SSIM harness for C8.
**Fitness function:** C1–C4 analytic tests.
**Done when:** a refraction/focal-point regression fails a unit test.

### T7 — Determinism + JNI fault-injection test kinds (Medium, S ~1 d) — Finding 8

**Problem:** The test net is broad in unit/integration/visual KINDS but missing render-determinism
and JNI-seam fault injection.
**Do:** (a) render the canonical scene twice and assert byte-identical PNG; (b) force the
`instanceId = -1` mid-frame failure path and assert graceful handling.
**Fitness function:** the determinism test (cheap; locks down the reproducibility arc42 §10 claims).
**Done when:** a non-deterministic render or an unhandled `-1` id fails CI.

### T10 — Fast-path regression guard (Medium, S 3–5 h) — Positive Pattern 1

**Problem:** The upload-once 4D-animation fast path (a strength of the design) is correctness-tested
but unprotected against silent regression to per-frame rebuild (a one-line condition flip).
**Do:** Add a counter on the renderer; assert an N-frame 4D animation issues
O(frames × instances) projection calls and O(1) instance builds.
**Fitness function:** the call-count assertion.
**Done when:** a regression to per-frame rebuild fails a test.

### T11 — Record OptiX-as-sole-backend as a deliberate ADR (Low, XS ~30 m) — backend coupling

**Problem:** The renderer backend is hard-wired to OptiX across ~9 scene builders (`OptiXRenderer`
as a concrete type, no `RenderBackend`/`SceneSink` interface). This is deliberate (AD-16) but
undocumented as a cross-cutting evolvability cost.
**Do:** Add a short note to arc42 §9 recording that a second backend would be a cross-cutting
refactor, not a plug-in. No code action unless a second backend is on the roadmap.
**Done when:** the decision is explicit in §9.
