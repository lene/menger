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

## Flaky test policy

Some integration tests are known to be intermittently flaky under GPU contention.
The current known-flaky list:

| Test | Reason | Policy |
|------|--------|--------|
| `sponge-volume` integration scenario | GPU contention under parallel runner load | One push retry before investigating |
| `tesseract-with-material` integration scenario | GPU contention under parallel runner load | One push retry before investigating |
| `Project4DGpuSuite` "animate 4D rotation faster via updateMesh4DProjection than via rebuild" | Zero-tolerance relative-timing assertion (`updateMs should be < rebuildMs`) between two `measureMs` blocks in the same process, no warm-up; thermal/load variance on the shared laptop GPU can invert the margin | Rerun the suite in isolation 2-3x (`sbt "mengerApp/testOnly io.github.lene.optix.Project4DGpuSuite"`) before treating a single failure as a regression |

**Retry limit:** A test may stay on the flaky list for at most **2 consecutive sprints**.
After that, root-cause investigation is mandatory before the next sprint starts.
Add to `CODE_IMPROVEMENTS.md` when a test goes onto the retry list.

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
3. If the failing (or passing) test compares two measured durations against each other with
   no tolerance (see the flaky-test table above), rerun that suite alone 2-3 times before
   trusting either a red or a green result.
4. Avoid running heavy concurrent GPU/build work while someone else might be independently
   verifying or pushing on the same shared machine — resource contention biases exactly this
   class of test.
