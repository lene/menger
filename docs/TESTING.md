# Test Failure Protocol

When a test fails, the default assumption is that **the test is catching a real bug**.
Do not rewrite or disable a test without completing this protocol.

---

## Investigation order

1. Run the failing test alone and read the actual failure message.
2. Check the test's git history (`git log -p -- <test-file>`) and recent changes
   to the code under test.
3. Decide which is wrong: implementation, test expectation, or both.
4. Document the investigation in the commit message.

---

## Decision tree

```
Test fails
    │
    ├─ Did a recent commit change the tested code?
    │       ├─ YES → Is the new behaviour correct?
    │       │           ├─ YES → Update the test expectation.
    │       │           │         Commit with: Test-Change: expectation changed because <reason>
    │       │           └─ NO  → Fix the implementation.
    │       └─ NO  → Is the test expectation still correct?
    │                   ├─ YES → Find the regression. Bisect if needed.
    │                   └─ NO  → Investigate whether the invariant itself changed.
    │
    └─ Red flags — stop and ask the user (see below)
```

---

## Red flags that require asking the user

Do not proceed autonomously when:

- Multiple tests fail after a "simple" refactor.
- Visual (integration) tests fail but unit tests pass.
- Only *some* similar tests fail (suggests an invariant that partially changed).
- A long-stable test starts failing with no relevant code change in recent history.

---

## Commit message format

When modifying a test, the commit message must carry a `Test-Change:` trailer
explaining why the expectation changed:

```
fix: correct sphere radius calculation

The radius was off by a factor of 2 due to diameter/radius confusion.

Test-Change: updated RadiusTest to expect correct 0.5 instead of 1.0
```

This trailer is **required by the `check-test-justification.sh` pre-push hook**
for any commit that modifies or deletes existing test files. Newly added tests
do not require it.

---

## Guard-proof rule: a new guard must fail against its defect (revert-and-run)

A new regression/guard test is only proof once demonstrated to fail against the defect it
guards. A guard that has never failed is unproven — it may be passing by accident (wrong
assertion, wrong fixture, a code path it never actually exercises).

**Procedure:** write the guard against the fixed code (it should pass). Temporarily revert
the fix — `git stash`, check out the pre-fix version of the file under test, or comment out
the fix — and re-run the exact same guard command. It must fail. Restore the fix, re-run
once more to confirm green, and record both results in the commit message.

### Worked examples (2026-08-07)

**OptiXException constructor test** (optix-jni, `efbb8a8`,
`JniErrorSurfaceSuite.scala:58-65`) — a reflection check that
`classOf[OptiXException].getConstructor(classOf[String])` exists and constructs correctly,
mirroring what JNI `ThrowNew` does at runtime. From the commit message: "verified it fails
with NoSuchMethodException on the pre-fix class and passes after."

**gas_registry alias check** (optix-jni, `1ad1d3a`, `GpuLeakSuite.scala:179-187`) — a
source-level check that `OptiXWrapper.cpp` never contains the aliasing expression
`gas_registry[static_cast<GeometryType>(-(instanceId + 1))]`. From the commit message:
"Verified: 42 occurrences before, 0 after, on the same command; with the fix reverted the
new check fails and the count returns to 43."

Both guards were run against pre-fix source before being trusted — that is the bar every
new guard test in this repo must clear.

---

## Flaky test policy

Some integration tests are known to be intermittently flaky under GPU contention.
The current known-flaky list:

| Test | Reason | Policy |
|------|--------|--------|
| `sponge-volume` integration scenario | GPU contention under parallel runner load | One push retry before investigating |
| `tesseract-with-material` integration scenario | GPU contention under parallel runner load | One push retry before investigating |

**Retry limit:** A test may stay on the flaky list for at most **2 consecutive sprints**.
After that, root-cause investigation is mandatory before the next sprint starts.
Add to `CODE_IMPROVEMENTS.md` when a test goes onto the retry list.

## Performance gates

Every timing-based assertion is a *gate* tagged `Perf` (`io.github.lene.qa.Perf`), built on the
shared `io.github.lene.qa.RelativeBenchmark` helper (vendored into `standards/test/scala/` from
menger-toplevel's `shared/standards/test-scala/`; edit it there, then `./bootstrap.sh sync`):

- A gate times a **subject** against a **reference** workload in interleaved rounds (warm-up,
  then 15 rounds alternating order), so throttling and background load affect both sides of
  every round alike. Rendering compares against a trivial scene through the same path (e.g.
  level-0 cube vs sponge); CPU work compares against a fixed CPU probe and uses
  `BenchConfig.JvmCpu` (GC before each sample, longer batches, longer warm-up).
- The verdict comes from the median of the per-round ratios and a sign-test confidence
  interval: PASS if the whole interval is within the limit, FAIL if the whole interval is
  beyond it, INCONCLUSIVE (test canceled) otherwise or when the interval is too wide.
- Limits are ~2x the highest upper confidence bound measured on the RTX A1000 laptop (also
  the CI runner), idle and under heavy CPU+GPU load; "X faster than Y" gates use 1.0. Each
  constant records its measurement.
- `menger-app/build.sbt` excludes the `Perf` tag from every test run; the `perf` suite
  (`scripts/suites/perf.sh`, push tier, right after `unit`) runs only that tag, alone:
  `PERF_ONLY=1 sbt "mengerApp/testOnly *"`. A canceled gate makes the suite SKIP with the gate
  named, never FAIL; the log line `PERF-INCONCLUSIVE ...` shows the measurement.
- The cross-commit trend check against `scripts/perf-baseline.json` is the release-tier
  `perf-trend` suite (`scripts/benchmark.sh`, same concept: each scene bracketed by calibration
  renders, baseline stored as machine-independent ratios).

---

## Pre-push hook behaviour

The pre-push hook runs the full test suite and blocks push on failure.
A `WIP:` commit prefix bypasses the hook for work-in-progress pushes.

The hook also runs `check-test-justification.sh` (Phase 0) before any compilation,
rejecting pushes where test files were modified without a `Test-Change:` trailer.

### Verifying hook/test output before declaring pass or fail

The hook keeps running later phases (packaging, integration) even after an earlier phase
has already set a failing status — a later phase actively executing is not evidence that
an earlier phase passed. Before claiming a run is green (to a user, in a commit message,
or as grounds to push):

1. Grep the log for ScalaTest's own failure markers — `\*\*\* FAILED \*\*\*` and
   `Failed tests:` — not just the compiler's `^\[error\]` prefix (misses test failures
   entirely) and not just a final `Tests: succeeded N, failed 0` summary (may not exist if
   the run was truncated).
2. Find the explicit named-stage checkpoint (e.g. `sbt test: PASSED`/`FAILED`) if the hook
   prints one — don't infer it from later output existing.
3. Timing assertions live only in the `perf` suite (see "Performance gates" above), which
   reports an unjudgeable measurement as SKIP rather than FAIL; a `perf` FAIL is conclusive.
4. Avoid running heavy concurrent GPU/build work while someone else might be independently
   verifying or pushing on the same shared machine — resource contention biases exactly this
   class of test.
