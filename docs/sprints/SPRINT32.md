# Sprint 32: Architecture Hardening & Quality Infrastructure

**Sprint:** 32 — Architecture Hardening
**Status:** 📋 Planned
**Estimate:** ~65 hours
**Branch:** `feature/sprint-32`

---

## Goal

Close the architecture backlog from the Sprint 30 review, upgrade build tooling, and
add automated quality gates. Every task converts a convention-maintained invariant
into a fitness function — the sprint's theme is *guard everything*.

---

## Success Criteria

- [ ] sbt 2.0.1 builds and tests pass across all three repos
- [ ] Object-type dispatch uses a single registry table (no more triplicated if/else chains)
- [ ] PerfCheck CI job asserts P1/P2 performance budgets against committed baselines
- [ ] Script-parity test catches `type=<...>` drift between integration and manual test scripts
- [ ] ArchUnit native-binding rule uses module-path scoping; `menger.geometry` renamed
- [ ] Caustics analytic rungs C1-C4 implemented as unit tests
- [ ] Render-determinism test and JNI fault-injection tests pass in CI
- [ ] Fast-path regression guard for 4D animations
- [ ] OptiX-as-sole-backend documented as an architectural decision (ADR)
- [ ] All tests pass (≥2,356)

---

## Tasks

### Task 32.1: Dependency & Hygiene Updates

**Estimate:** 6h

**32.1a — sbt 2.0.1 upgrade (4h):**
- Bump `sbt.version` in `project/build.properties` across all three repos
- Fix any build definition API changes (sbt 2.0 removes deprecated syntax)
- Verify `sbt compile`, `sbt test`, `sbt "scalafix --check"` pass
- Update CI Docker image or document that sbt is auto-bootstrapped
- **Risk:** sbt 2.0 is a major release — may need plugin updates or build.sbt changes

**32.1b — CODE_IMPROVEMENTS.md cleanup (1h):**
- Verify Sprint 31 tasks 31.6 + 31.8 actually resolved the claimed items
- Remove resolved items from CODE_IMPROVEMENTS.md
- Add any new findings discovered during sprint-close

**32.1c — Record OptiX-as-sole-backend ADR — T11 (30m):**
- Add a short note to arc42 §9: the renderer is deliberately hard-wired to OptiX
  across ~9 scene builders; a second backend would require a cross-cutting refactor
- Document the decision, rationale, and when to revisit

### Task 32.2: Unify Object-Type Dispatch — T1

**Estimate:** 16h  
**Priority:** High  
**Finding:** F1 (from ARCHITECTURE_REVIEW.md 2026-06-12)

**Problem:** `ObjectType` dispatch is triplicated across `GeometryRegistry.builderFor`,
`RenderModeSelector`, and `InteractiveEngine.buildScene4DTrackedOrFallback`. Three
hand-maintained if/else chains that already drifted once (sierpinski4d/hexadecachoron4d
gap fixed in Sprint 27 but root cause remains).

**Implementation:**
- Introduce a single source of truth: a registry `Map` of
  `(predicate, builderFactory)` consumed by all three sites
- `InteractiveEngine` still supplies its recorder callbacks, but the
  *type→builder* decision lives in one place
- Refactor `GeometryRegistry` to own the dispatch table
- Remove the three if/else chains; every new type touches one table + one builder

**Fitness function:** The completeness test from the 2026-06-12 patch (every
`ObjectType.VALID_TYPES` entry resolves to exactly one builder) becomes the
guard over the unified table.

**Done when:** The three if/else chains are gone; adding a type touches only the
unified table and one new builder class.

### Task 32.3: Quality Infrastructure

**Estimate:** 17h

**32.3a — Performance governance — T2 (8h):**
- Wire the existing `scripts/benchmark.sh` + `PerfCheck` CI job to arc42 §10.4
  budgets (P1 <5s, P2 <500ms)
- Commit a `perf-baseline.json` with current measurements
- Add `PerfCheck` as a blocking CI job on MR pipelines
- Emit machine-readable results for trend tracking

**32.3b — Script-parity fitness function — T9 (3h):**
- Create a test that extracts `type=<...>` tokens from `integration-tests.sh`
  and `manual-test.sh`
- Assert `integration ⊇ manual` coverage over `ObjectType.VALID_TYPES`
- Fail CI on divergence (adding a type to only one script)

**32.3c — Determinism + JNI fault-injection — T7 (6h):**
- Render the canonical scene twice, assert byte-identical PNG output
- Force `instanceId = -1` mid-frame failure paths for sphere, mesh, curve,
  cylinder, and cone instance-adding JNI calls
- Verify `Try`→`Failure` propagation with scalamock

### Task 32.4: Architecture Hardening

**Estimate:** 10h

**32.4a — Module-scoped ArchUnit native rule — T5 (6h):**
- Make native-binding ArchUnit rules module-path-based (bindings originate only
  from expected module locations)
- `menger-geometry` classes extend the published optix-jni surface, never duplicate it
- Rename `menger.geometry` → `menger.video` (or assert it has no `@native` methods)
- **Fitness function:** module-path-scoped native-binding ArchUnit rule

**32.4b — Fast-path regression guard — T10 (4h):**
- Add a counter on the renderer tracking projection calls vs instance builds
- Assert an N-frame 4D animation issues O(frames × instances) projection calls
  and O(1) instance builds
- Catch regression to per-frame rebuild (a one-line condition flip that
  could silently degrade performance)

### Task 32.5: Caustics Ladder — T6

**Estimate:** 20h  
**Priority:** Medium  
**Finding:** F7

**Problem:** The C1-C8 caustics ladder is 0/8 implemented; no SSIM is computed.
`docs/caustics/CAUSTICS_TEST_LADDER.md` documents the ladder but nothing enforces it.

**Implementation:**
- Implement analytic rungs C1-C4 as `AnyFlatSpec` determinism tests (no GPU needed —
  they catch refraction-math regressions)
- C1: Refraction plane — ray through air-to-glass boundary; assert Snell's law
- C2: TIR angle — assert critical-angle behavior
- C3: Focal point — rays through a sphere converge; assert focal distance
- C4: Fresnel reflectance — assert R+T=1 at boundary
- Mark C5-C8 "not implemented" in arc42 §10 so the doc stops over-claiming
- Defer the SSIM harness for C8

**Fitness function:** C1-C4 analytic tests.

**Done when:** A refraction/focal-point regression fails a unit test.

### Task 32.6: 4D Hilbert Curve Preset

**Estimate:** 2h

- Add a proper 4D Hilbert curve preset to `LSystemPresets` (uses `>`,`<`
  ana-axis rotation symbols in the grammar)
- Generate reference image via integration test
- Add to both `integration-tests.sh` and `manual-test.sh`

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 32.1 | sbt 2.0.1 + CODE_IMPROVEMENTS cleanup + T11 ADR | 6h |
| 32.2 | T1 — Unify object-type dispatch | 16h |
| 32.3 | T2 (PerfCheck) + T9 (script parity) + T7 (determinism) | 17h |
| 32.4 | T5 (ArchUnit) + T10 (fast-path guard) | 10h |
| 32.5 | T6 — Caustics ladder C1-C4 | 20h |
| 32.6 | 4D Hilbert curve preset + ref image | 2h |
| **Total** | | **~71h** |

---

## Task Dependency Graph

```
32.1 (sbt + cleanup + ADR) ── independent, must go first (sbt impacts all)
    │
    ├─► 32.2 (T1 dispatch) ── independent of quality infra
    │
    ├─► 32.3a (T2 PerfCheck) ── independent
    │       │
    │       └──► 32.3b (T9 script parity) ── benefits from dispatch unification
    │
    ├─► 32.3c (T7 determinism) ── independent
    │
    ├─► 32.4a (T5 ArchUnit) ── independent
    │
    ├─► 32.4b (T10 fast-path) ── independent
    │
    ├─► 32.5 (T6 Caustics) ── independent
    │
    └─► 32.6 (4D Hilbert) ── independent, quick
```

## Parallelization Opportunities

| Pair | Why |
|------|-----|
| 32.2 + 32.3a | Dispatch unification and PerfCheck are independent code areas |
| 32.3b + 32.3c | Script parity and determinism tests — different files |
| 32.4a + 32.4b | ArchUnit rules and fast-path guard — different concerns |
| 32.5 + 32.6 | Caustics is CPU-only, Hilbert is GPU render — no contention |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
- [ ] Architecture backlog items marked done
- [ ] ARCHITECTURE_REVIEW.md updated with closure notes
