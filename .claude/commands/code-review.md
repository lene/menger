# Code Review (Scala + JNI + CUDA)

A review procedure for this project's stack: Scala 3 business logic, a JNI bridge, and C++/CUDA (including OptiX) on the native side.

The goal is **depth over breadth**. A short review that names two real architectural problems with concrete evidence is more valuable than a long review of style smells. You are not a linter — Scalafix, WartRemover, and clang-tidy already exist. Your job is what they can't do: reason across the seam, across files, and about the architecture as a whole.

## Output

Write findings to `CODE_IMPROVEMENTS.md`. See the "Reporting" section at the end. **Do not read it until step 4** — see "Why" in that section.

---

## Phase 0 — Run the deterministic tools first

Before reading any source code, run whatever's available. You are layering on top of these tools, not duplicating them.

```bash
# Scala side — run whichever exist in the project
sbt scalafixAll --check 2>&1 | tee /tmp/review-scalafix.log     || true
sbt 'set ThisBuild / scalacOptions += "-Wunused:all"' compile 2>&1 | tee /tmp/review-scalac.log || true
sbt scapegoat 2>&1 | tee /tmp/review-scapegoat.log              || true
# WartRemover usually runs as part of compile; capture its output above.

# ArchUnit — if the project has architecture tests, run them. They enforce
# package/layer rules, cycle detection, etc. as failing JUnit tests.
# Detect by searching for ArchUnit imports in test sources.
grep -rln 'com.tngtech.archunit' src/test 2>/dev/null && \
  sbt 'testOnly *Arch*' 2>&1 | tee /tmp/review-archunit.log || \
  echo "no ArchUnit tests found" | tee /tmp/review-archunit.log

# Native side
find . -name '*.cpp' -o -name '*.cu' -o -name '*.h' | head -1 >/dev/null && {
  # If a compile_commands.json exists, prefer it; otherwise skip and note it.
  test -f compile_commands.json && \
    clang-tidy -p . $(git ls-files '*.cpp' '*.cu') 2>&1 | tee /tmp/review-clangtidy.log || \
    echo "no compile_commands.json — clang-tidy skipped" | tee /tmp/review-clangtidy.log
  cppcheck --enable=warning,performance,portability --inline-suppr \
    $(git ls-files '*.cpp' '*.h') 2>&1 | tee /tmp/review-cppcheck.log || true
}

# Sizes — used later, not as findings themselves
git ls-files '*.scala' '*.cpp' '*.cu' '*.h' | xargs wc -l 2>/dev/null | sort -rn > /tmp/review-sizes.txt
```

If a tool is missing or not configured, note it in the report under "Tooling gaps" and move on. **Do not re-implement what these tools do.** Anything they already flag goes into a single summary line in the report (e.g., "Scalafix: 47 violations, see scalafix output") — not as individual findings.

If a tool fails to run for environmental reasons (missing `compile_commands.json`, no GPU, sbt task missing), that's itself a finding: the project has gaps in its deterministic review layer.

---

## Phase 1 — Map the architecture (don't review yet)

Spend the first pass building a mental model, not finding issues. Write nothing to the report yet.

```bash
# Module layout
git ls-files | sed 's|/[^/]*$||' | sort -u | head -50

# Scala package structure
git ls-files '*.scala' | sed 's|/[^/]*\.scala$||' | sort -u

# JNI surface — both sides
grep -rn 'extern "C"' --include='*.cpp' --include='*.h' .
grep -rn '@native' --include='*.scala' .
grep -rn 'JNIEXPORT\|JNICALL' --include='*.cpp' --include='*.h' .

# CUDA kernel launches and the OptiX surface
grep -rn '<<<' --include='*.cu' --include='*.cpp' .
grep -rn 'optix\(Launch\|PipelineCreate\|ProgramGroupCreate\)' --include='*.cpp' --include='*.cu' .

# Resource lifetimes
grep -rn 'cudaMalloc\|cudaFree\|cuMemAlloc' --include='*.cpp' --include='*.cu' --include='*.h' .
grep -rn 'NewGlobalRef\|DeleteGlobalRef\|NewLocalRef' --include='*.cpp' .
```

Identify:
- The **JNI surface area**: which Scala classes have `@native` methods, which C++ functions implement them. List them.
- The **ownership model**: who allocates GPU memory, who frees it, how that responsibility crosses the JNI boundary.
- The **error-propagation path**: how a `cudaError_t` or OptiX error becomes something Scala sees.
- The **threading model**: are JNI calls made from one thread or many? Is the CUDA context per-thread or shared?

Write this map down in scratch (your own notes, not the report). It is the frame for everything that follows.

---

## Phase 2 — Review the seams (this is where the value is)

Use the `Task` tool to spawn focused sub-agents in parallel, one per seam. Each gets a narrow remit and a small set of files. Do **not** ask any sub-agent to "review the codebase."

### Seam 1 — The JNI boundary

This is the single highest-value review surface and the one no linter checks well. Focused checklist:

**On the C++ side, for every JNI function:**
- After every JNI call that can throw (`Find*`, `Get*`, `Call*`, `NewObject`, `*ArrayElements`, etc.) — is there an `ExceptionCheck`/`ExceptionOccurred` before the next JNI call or before returning to Java? Missing checks here cause subtle, hard-to-debug crashes much later.
- Every `NewGlobalRef` paired with `DeleteGlobalRef`? Every `NewLocalRef` either deleted explicitly or known to fit in the local-frame budget (default 16)? Long-running native code without `PushLocalFrame`/`PopLocalFrame` is a smell.
- `GetPrimitiveArrayCritical` / `GetStringCritical` regions: are they short, free of JNI calls, free of blocking? Calling back into Java inside a critical region can deadlock.
- Mode argument to `Release*ArrayElements` — using `0` vs `JNI_ABORT` vs `JNI_COMMIT` correctly? Wrong mode = silent data loss or wasted copies.
- Thread attachment: any native thread calling into Java without `AttachCurrentThread` / `DetachCurrentThread`?
- Pinned vs. unified memory: if `cudaHostAlloc`/`cudaMallocManaged` is used, who owns the lifetime, and is that lifetime tied to Java GC behavior in a way that could surprise?

**At the Scala/C++ boundary contract:**
- Are native methods returning raw `Int` error codes that Scala ignores, or are errors translated into Scala's error type? (The skill memory mentions "Error propagation: consistent error handling between JNI boundary" — verify this is actually true, don't trust it.)
- Are GPU resource handles (kernel handles, OptiX pipelines, device pointers) modeled in Scala in a way that survives GC unpredictability? A `Long` field holding a `CUdeviceptr` with no `Closeable`/`Using` discipline is a leak waiting to happen.
- Is there a single ownership model, or do different parts of the codebase use different conventions? Inconsistency here is worse than either convention alone.

**Specific things to grep for:**
```bash
# JNI calls without subsequent ExceptionCheck — rough heuristic
grep -n 'env->\(Find\|Get\|Call\|New\)' src/**/*.cpp | head -50
# Then for each, check the next ~10 lines for an exception check.

# Possible global ref leaks
grep -c NewGlobalRef src/**/*.cpp
grep -c DeleteGlobalRef src/**/*.cpp
# Counts should be in the same ballpark, modulo program-lifetime singletons.
```

### Seam 2 — CUDA correctness and resource lifetime

For each kernel launch site and each device allocation:

- **Launch error handling**: is `cudaGetLastError()` called after `<<<...>>>`? Is `cudaDeviceSynchronize()` (or stream sync) used where the next operation depends on completion? Common bug: kernel launch fails silently, the next `cudaMemcpy` "succeeds," and you get garbage data.
- **Stream discipline**: are all operations on the same data on the same stream, or properly synchronized between streams?
- **Memory lifetime**: every `cudaMalloc` paired with `cudaFree` on every exit path including exceptions. RAII wrappers in use? If not, that's a finding.
- **OptiX pipelines / programs / SBT records**: lifetime tied to what? Reloaded how? Are there leaks across reloads (common during dev iteration).
- **Shared memory bank conflicts, register pressure, occupancy** — only flag if you can see evidence; don't speculate. If Nsight Compute output exists, read it.
- **Determinism**: any use of atomics on floats, non-deterministic reductions, or undefined ordering that would make results irreproducible? Often invisible in code review until it bites.

If `compute-sanitizer` output is available (memcheck, racecheck, initcheck, synccheck), summarize it in one block — don't restate it as individual findings.

### Seam 3 — Scala architecture and FP discipline

**Skip everything WartRemover and Scalafix already cover.** No findings for `var`, `null`, `asInstanceOf`, `while`, missing explicit types — these are linter territory. If they appear in code, that means the linter isn't running or is misconfigured: report that as one finding, not many.

What to actually review:

- **Module dependencies**: does the package structure express the intended architecture? Are there cycles? Does the JNI-facing module leak its types upward into pure business logic?

  *If the project has no ArchUnit (or equivalent) tests enforcing these rules, that's itself a finding under "Tooling gaps" — architectural invariants maintained by review are unreliable; they should be tests that fail builds. ArchUnit works on Scala bytecode and can express cycle detection, layer dependencies, and "module X must not depend on module Y" rules as failing JUnit tests. The Scala-side API is awkward (it's a Java fluent DSL) but the rules survive Scala/Java/Kotlin mixing because it operates on bytecode.*
- **Effect discipline**: where do effects live (IO, Future, Try, Either)? Are they pushed to the edges or scattered? Is there a single error type or many ad-hoc ones?
- **Type-level expression of invariants**: are GPU resource handles, color spaces, coordinate systems, etc. distinguished at the type level (opaque types, phantom types, refined types), or all `Double`/`Long`? "Consistent color representation" is the kind of thing that is easy to claim and hard to enforce without types.
- **Abstraction match**: places where the same concept is re-implemented (e.g., several ad-hoc resource-management patterns instead of one `Resource[F, A]` or `Using` pattern).
- **The shape of the public API**: what does a caller have to know to use this correctly? If the answer is "a lot of conventions that aren't in the type signature," that's a finding.

### Seam 4 — Cross-cutting concerns

One pass, deliberately looking for patterns that span files:

- **Error handling**: count the distinct ways errors are represented. If it's >2, that's a finding.
- **Logging**: consistent? Or `println` in one place, SLF4J elsewhere, swallowed exceptions in a third?
- **Configuration**: where do magic numbers live? Compiled in, or in a config? Are GPU-specific tunables (block size, stream count, batch size) discoverable?
- **Resource cleanup discipline**: is there one pattern (`Using`, `Resource`, try-finally) or several?
- **Test coverage gaps**: not coverage *percentage* — gaps in *kinds* of tests. Is there anything that exercises the JNI boundary under load, fault injection, or long-running scenarios?

### Seam 5 — Recent changes (only if a diff is meaningful)

If `git diff origin/main...` shows substantive changes, do a targeted pass on them. But this is the *last* seam, not the first — anchoring on recent code is exactly the trap that makes reviews superficial. Often the most important architectural issues predate any specific PR.

---

## Phase 3 — Synthesize

Now collect findings from all seams. Apply the following filters in order:

1. **Drop anything a linter would catch.** If WartRemover would flag it, it doesn't belong here.
2. **Drop anything you can't ground.** Every finding needs a `file:line` reference *and* a concrete example. "Possible coupling issue in the renderer module" is not a finding; "RenderingContext.scala:142 reaches into PipelineState's internal `_kernels` field, bypassing the public API, in 4 places" is.
3. **Merge related findings.** Three instances of the same pattern is one finding ("inconsistent JNI error handling, 3 instances: A.cpp:12, B.cpp:88, C.cpp:201"), not three.
4. **Rank by architectural blast radius**, not by line count. A single missing `ExceptionCheck` in JNI code is worse than a 600-line class.

Aim for **5–15 findings total**. If you have more than 15, you're either reporting linter-territory items or not merging. If you have fewer than 3, either the codebase is in great shape (say so) or you didn't look hard enough (re-examine the seams).

---

## Phase 4 — Reconcile with prior review

**Now**, and not before, read `CODE_IMPROVEMENTS.md` if it exists.

**Why this ordering**: reading prior findings first anchors the review on yesterday's agenda. A finding that survives a no-prior-context review is, by construction, one a fresh reviewer would flag. That makes the issue list a stronger signal over time.

Reconciliation:

- **Overlap** between a new finding and an old one → merge. Keep the new finding's framing (it's based on current code); incorporate any context from the old one (history of the issue, prior decisions).
- **Old finding no longer appears in your new review** → it's a candidate for "Resolved." Before declaring resolution, look at the specific file:line from the old finding and verify the pattern is actually gone, not just moved. If unsure, mark as "Likely resolved — verify."
- **Old finding still real but didn't make your priority list** → carry forward under "Carried forward." This is normal; not everything gets addressed every sprint.
- **New finding** → just include it.

---

## Reporting — `CODE_IMPROVEMENTS.md`

Renumber findings each review from 1, ordered by priority. Do not preserve old IDs across runs — file:line + title is the stable identifier, not the ID.

```markdown
# Code Quality Review — YYYY-MM-DD

## Summary

2–4 sentences. Most important architectural observation, overall health, biggest single
risk. No buzzwords. Reader should know whether to keep reading after this paragraph.

## Tooling status

- Scalafix: ran / not configured / N violations (see scalafix log)
- WartRemover: ran / not configured / N violations
- Scapegoat: ran / not configured / N warnings
- ArchUnit: N rules, all passing / N rules, M failing / no rules defined
- clang-tidy: ran / no compile_commands.json / N warnings
- cppcheck: ran / N warnings
- Compute Sanitizer: results from last run available / not run
- Tooling gaps: <anything missing that should be wired up>

## Findings

### 1. <Concise title naming the actual problem>

**Where**: `path/file.scala:120`, `path/file2.cpp:88`
**Impact**: Critical / High / Medium / Low — and one sentence on why this severity.
**Effort**: rough order of magnitude (hours / days / weeks)

**What**: Specific description with concrete evidence. Quote 1–5 lines of code if it
clarifies. No generic definitions of code smells.

**Why it matters**: The actual consequence — not "reduces maintainability" but
"every new kernel launch site has to re-implement the same five-line error check,
and three of the existing seven have it subtly wrong."

**Suggested direction**: Not a full refactoring plan — the direction. "Introduce a
single `withCudaErrorCheck` combinator and migrate callers" is enough. The team
will figure out the rest.

### 2. ...

## Carried forward from prior review

Items still present but not in this review's priority set. One line each with
file:line reference.

## Resolved since last review

Items from prior review no longer present. One line each with brief verification
note ("verified — `RenderingContext` no longer exposes `_kernels`").

## Positive patterns worth preserving

0–3 items, only if there's something genuinely worth pointing out. Don't pad.
```

### Severity calibration

- **Critical**: causes correctness bugs, data loss, or memory safety issues (especially at the JNI/CUDA seam). Or blocks a known upcoming change.
- **High**: significantly slows development of every new feature in this area, or makes the system noticeably hard to reason about.
- **Medium**: real problem but localized; will become high if not addressed within a couple of months.
- **Low**: real but minor; address opportunistically. Don't generate Low findings just to fill space — if a finding is genuinely Low, consider whether it belongs in the report at all.

### What does *not* belong

- Style issues a linter catches.
- Generic restatements of clean code or SOLID principles without a specific instance.
- Speculative concerns ("this *might* become a problem if you ever do X").
- Findings without a file:line reference.
- Aesthetic preferences disguised as architectural concerns.
- Long lists of "Long Method" or "Large Class" findings — pick the worst 1–2 instances if size is genuinely the problem; otherwise size alone isn't a finding.

---

## Project-specific notes

**Out of scope as findings, but in scope for understanding:**
- OptiX/CUDA code using imperative style — this is correct for the domain. Only flag if the *Scala-side* code mimics it without need.
- JNI code using imperative style and explicit lifetime management — same.
- Test code using mutable setup — acceptable.

**Specifically in scope and high-value:**
- Anything at the JNI seam (error handling, ref management, ownership, threading).
- GPU resource lifetime visible from Scala.
- Type-level expression of GPU/rendering invariants (color spaces, coordinate frames, handles).
- Consistency of error propagation across the seam.
- Any place where Scala code makes assumptions about native-side state that aren't enforced by types.

---

## A note on uncertainty

Findings vary in confidence. Use these labels sparingly, only when relevant:

- **(certain)** — the issue is unambiguous; no judgment call.
- **(judgment)** — the issue is real but reasonable engineers might prioritize differently.
- **(needs verification)** — the pattern looks wrong but the codebase context might justify it; check with the team before acting.

Default to no label. Apply only when the call is genuinely contested.
