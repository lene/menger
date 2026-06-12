# Architectural Review Command (`/arch-review`) — Design

**Date:** 2026-06-12
**Status:** Approved (brainstorming) — pending implementation plan
**Author:** brainstorming session (Lene + Claude)

## 1. Problem

The project has `/code-review` (`.claude/commands/code-review.md`), which does **seam-level
correctness** review: JNI exception checks, CUDA resource lifetime, FP discipline, file:line
findings. It is excellent at "is this code correct across the JNI/CUDA seam."

It does **not** answer the higher-level question: *is the architecture sound, mature, and able
to absorb change — and are its invariants enforced or merely hoped for?*

We want a second, deliberately **disjoint** review focused on architecture as a whole, tuned to
this specific project (a Scala 3 / JNI / CUDA / OptiX ray tracer), not a generic enterprise
checklist.

## 2. Decisions (locked during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Form factor | **Command** `/arch-review` | Deliberate invocation, parity with `/code-review`. No auto-trigger. |
| Output | **`ARCHITECTURE_REVIEW.md`** (repo root) | Mirrors `CODE_IMPROVEMENTS.md`; reconciled across runs. |
| Execution | **Parallel subagents, one per axis** | Matches `/code-review`'s seam-parallel design; depth per axis. |
| "Mobility" means | **Evolvability** | User scoped it explicitly (not portability/deployability). |
| Approach | **B — Report + fitness-function nomination & audit** | Turns a static report into governance; fits a project already governed by the pre-push hook. |
| Axes | **4: Soundness, Maturity, Evolvability, Performance Architecture** | First three are generic-but-grounded; the 4th reflects the project's #1 declared quality goal. |

## 3. Guiding frame: evolutionary architecture / fitness functions

The intellectual spine is **fitness functions** (Ford/Parsons/Kua, *Building Evolutionary
Architectures*) plus the **Modularity Maturity Index**. The differentiator from a static report:

For every structural finding, the review delivers a **verdict on enforcement**:
- **Guarded** — an existing automated check (test, CI gate, ArchUnit rule, coverage ratchet,
  version-consistency check, memory-leak gate) already protects this invariant. *Credit it.*
- **Partially guarded** — a fitness function exists but its coverage has gaps. *Name the gap.*
- **Unguarded** — the invariant is maintained only by convention/review. *Nominate* a concrete
  fitness function (e.g. an ArchUnit rule, a dependency test, a change-cost metric).

The review **does not write** the fitness-function code (that is a follow-up implementation task,
Approach C, explicitly out of scope here). It nominates and audits.

**Important:** ArchUnit is **already present** (`menger-app/src/test/scala/menger/ArchitectureSpec.scala`,
`ArchitecturePhase2Spec.scala`, design `docs/superpowers/specs/2026-05-21-archunit-rules-design.md`).
The Soundness axis therefore **audits the coverage** of the existing rules against the invariants it
finds — it does not naively propose rules that already exist.

## 4. Disjointness contract with `/code-review`

To justify a second review, the two must not overlap. Hard boundaries:

**`/arch-review` does NOT report (belongs to `/code-review` or linters):**
- Per-call JNI exception-check omissions, ref leaks, `Release*ArrayElements` mode bugs.
- Per-site CUDA error-check / sync / `cudaMalloc`–`cudaFree` pairing.
- WartRemover/Scalafix territory (`var`, `null`, `asInstanceOf`, line length).
- Any single-file, single-line correctness defect.

**`/arch-review` DOES report (its exclusive remit):**
- Module dependency direction, cycles, boundary integrity (whole-system).
- Whether invariants are enforced by fitness functions vs convention.
- Change-cost / extension-seam quality.
- Whether architecture matches arc42, and whether structure protects the performance budget.

Rule of thumb: if a finding has a single `file:line`, it is probably `/code-review`'s. If it spans
modules or asks "is this guarded," it is `/arch-review`'s.

## 5. The four axes

Each axis is run by one focused subagent with a narrow remit and is **anchored to a specific arc42
section**, which it cross-validates against the actual code (turning "doc drift" into a built-in
check rather than a separate axis).

### Axis 1 — Soundness (structure is correct) · anchor: arc42 §5 Building Blocks

- Module dependency **direction**: does `menger-common` stay pure (no `optix-jni` / LibGDX / JNI
  types leaking upward)? Does `optix-jni` leak native types into `menger-common` / `menger-app`?
- **Cycles** between modules/packages.
- **Boundary integrity**: the three module seams — does `menger-app` depend only on stable
  interfaces of `menger-common` / `optix-jni`?
- **Cohesion / god-objects**: e.g. `InteractiveEngine` accreting render + stats + save + exit.
- **Type-level invariants**: dimensions (3D/4D), color spaces, coordinate frames, GPU handles
  modeled as types vs raw `Double`/`Long`.
- **Trust-boundary input validation** (folded-in security sub-point): CLI / scene-file parsing
  validates before use (cf. `M-sceneb-validate-bypass`).
- **arc42 §5 drift**: does the documented building-block view match the real module graph?
- **Fitness-function audit**: do `ArchitectureSpec` / `ArchitecturePhase2Spec` rules actually cover
  the dependency/cycle invariants found here? Name uncovered invariants.

### Axis 2 — Maturity (structure is governed & robust) · anchor: arc42 §10 Quality, §11 Risks

- **Error model**: one error representation across the seam, or many ad-hoc ones? Is there a
  strategy, not just instances.
- **Test-architecture *shape*** (not coverage %): the *kinds* of safety net — unit / integration /
  visual-regression / memory-leak / determinism. Gaps in kind, e.g. JNI-boundary fault injection.
- **Correctness reproducibility** (folds in priority #2): canonical scenes, the caustics
  analytic→statistical→image ladder, render determinism — are correctness claims *governed*?
- **Observability**: logging strategy coherence; `--stats-json` / `benchmark.sh` as signals.
- **Tech-debt aging**: `CODE_IMPROVEMENTS.md`, arc42 §11 — are debts tracked and decreasing or
  accreting?
- **Governance maturity**: credit existing fitness functions (coverage ratchet ≥80%, version
  consistency across 4 files, Valgrind/compute-sanitizer gates, pre-push hook).
- **arc42 §10/§11 drift**: do stated quality scenarios / risks reflect reality?

### Axis 3 — Evolvability (structure absorbs change) · anchor: arc42 §9 Decisions

- **Change-cost of representative changes**, measured in files/modules touched:
  - add a new fractal/geometry type (cf. fitness function `M4: < 1 day`),
  - add/extend a dimension (3D → 4D generalization — parameter or duplicated paths?),
  - add a material / shader path,
  - swap or add a render backend.
- **Backend-portability seam**: is NVIDIA OptiX hard-wired throughout, or abstracted behind an
  interface (cf. `docs/architecture/optix-jni-decoupling.md`)? Evolvability, not portability-as-goal.
- **LibGDX lock-in** of windowing/UI.
- **Open/closed at architecture level**: do new features plug into seams, or force edits to stable
  core?
- **Known smell to quantify**: CLAUDE.md already mandates a new rendering feature be added to
  `integration-tests.sh` **and** `manual-test.sh` **and** the shader path — surface and measure
  this shotgun-surgery cost; nominate a parity fitness function if unguarded.
- **arc42 §9 drift**: are recorded ADRs still honored by the code?

### Axis 4 — Performance Architecture (structure enables & protects speed) · anchor: arc42 §10.4 Budgets

Scoped to performance **as a structural property** — explicitly **not** raw benchmarking
(`benchmark.sh` owns that), the same way the review stays disjoint from `/code-review`.

- Does an abstraction layer sit in the **hot path** (per-ray / per-pixel / inner kernel) and tax it?
- Does the module split (`common` / `app` / `optix-jni`) let the hot path stay hot, or force
  crossings of the JNI seam in tight loops?
- Algorithmic-complexity decisions visible in structure (surface subdivision `O(12^n)` vs
  `O(20^n)` — is the efficient choice structurally locked in or easy to regress?).
- **Is the performance budget a fitness function?** Credit `PerfCheck` CI + §10.4 budgets +
  `benchmark.sh` baselines; identify which budgets are *advisory* vs *enforced* and nominate
  promotion where a regression would be expensive.
- GPU tunables (block size, stream count, batch size) discoverable/changeable without recompile?
- **arc42 §10.4 drift**: do documented budgets match measured baselines?

## 6. Considered and rejected axes

- **Security** — local CLI render tool; no network/auth/multi-tenant. Only real surface is JNI
  memory safety (owned by `/code-review`) + input validation (folded into Soundness). Not an axis.
- **Deployability / Portability** — real complexity (OptiX SDK↔driver matching, CUDA versions,
  headless `xvfb`, Docker) but user scoped mobility→evolvability and deselected these. A note at most.
- **Usability / CLI ergonomics** — product concern, not architecture.
- **Observability / Correctness fidelity** — folded into Maturity rather than peer axes.

## 7. Execution flow

```
Phase 0 — Map (no findings yet)
  • module layout, package graph, inter-module deps
  • JNI surface (@native ↔ extern "C"), backend seam
  • inventory existing fitness functions: ArchUnit specs, coverage ratchet,
    version check, PerfCheck, memory-leak gates, integration/manual scripts
  • cross-reference arc42 §5/§9/§10/§11 to the real tree
Phase 1 — Spawn 4 parallel subagents (Task tool), one per axis.
  Each gets: narrow remit, its arc42 anchor, the Phase-0 map, the fitness-function
  inventory, and the disjointness contract (§4). None may "review the codebase."
Phase 2 — Synthesize: collect, dedupe, merge same-pattern findings, rank by
  architectural blast radius (not line count).
Phase 3 — Reconcile with prior ARCHITECTURE_REVIEW.md (resolved / carried-forward /
  new), verifying resolutions at the cited location.
Phase 4 — Write ARCHITECTURE_REVIEW.md.
```

Filters before reporting (from `/code-review`, adapted): drop anything a linter or `/code-review`
would catch; drop anything not grounded in module/file evidence; merge related; rank by blast
radius. Target **5–15 findings**.

## 8. Report format — `ARCHITECTURE_REVIEW.md`

```markdown
# Architectural Review — YYYY-MM-DD

## Summary
2–4 sentences: biggest structural observation, overall architectural health,
single largest evolvability/maturity risk.

## Axis scorecard
| Axis | Health | Headline finding |
|------|--------|------------------|
| Soundness            | green/yellow/red | ... |
| Maturity             | ...              | ... |
| Evolvability         | ...              | ... |
| Performance Arch.    | ...              | ... |

## Fitness-function status
| Invariant | Guarded by | Status | Action |
|-----------|-----------|--------|--------|
| no menger-common → optix-jni dep | ArchitectureSpec | guarded | — |
| perf budget P2 (<500ms)          | PerfCheck (advisory) | partial | nominate: make enforced |
| add-geometry parity (3 scripts)  | none | unguarded | nominate: parity test |

## Findings
### N. <title naming the actual structural problem>
**Axis:** Soundness | Maturity | Evolvability | Performance
**Where:** module(s) / `path:line` evidence
**Impact:** Critical/High/Medium/Low + one sentence
**Effort:** hours / days / weeks
**Enforcement:** guarded / partially guarded / unguarded
**What:** concrete, evidenced.
**Why it matters:** real consequence, not "reduces maintainability."
**Suggested direction:** the direction, not a full plan. If unguarded, the nominated
  fitness function.

## arc42 coherence
Per-section drift notes (§5, §9, §10, §11) — doc vs reality.

## Carried forward from prior review
## Resolved since last review
## Positive patterns worth preserving (0–3, no padding)
```

Severity calibration and "what does not belong" are inherited verbatim from `/code-review`
(`.claude/commands/code-review.md` §"Severity calibration" / §"What does *not* belong"), minus the
seam-correctness items that are now explicitly `/code-review`'s remit.

## 9. Deliverable & location

- New command file: **`.claude/commands/arch-review.md`** (the implementation of this design).
- It references this spec for rationale but is self-contained operationally.
- No change to `/code-review`; the two are complementary and disjoint by §4.

## 10. Out of scope

- Auto-generating ArchUnit rules / test stubs (Approach C) — separate future task.
- Any change to `benchmark.sh`, CI, or the pre-push hook.
- Deployability/portability review.

## 11. Open questions

- None blocking. Health-rating scheme (green/yellow/red) is a presentation detail, adjustable
  during implementation.
