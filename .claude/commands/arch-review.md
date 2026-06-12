# Architectural Review (`/arch-review`)

A whole-system architecture review for this project: Scala 3 domain logic, a JNI bridge, and
C++/CUDA/OptiX on the native side. Four axes — **Soundness, Maturity, Evolvability, Performance
Architecture** — each anchored to an arc42 section and judged through the lens of **fitness
functions**: is each invariant *enforced* by an automated check, or merely hoped for?

Design rationale: `docs/superpowers/specs/2026-06-12-arch-review-design.md`.

## What this review is — and is not

This sits **one level above** `/code-review`. The two are deliberately **disjoint**.

- `/code-review` owns seam-level correctness: per-call JNI exception checks, ref leaks, per-site
  CUDA error/sync/`cudaMalloc`-`cudaFree` pairing, linter territory. **Do not report these here.**
- `/arch-review` owns whole-system structure: module dependency direction, cycles, boundary
  integrity, change-cost, whether invariants are guarded by fitness functions, arc42 coherence,
  and whether structure protects the performance budget.

Rule of thumb: a finding with a single `file:line` is probably `/code-review`'s. A finding that
spans modules or asks *"is this guarded?"* is this review's.

**You are not a linter.** Scalafix, WartRemover, clang-tidy, ArchUnit already exist. Your value is
reasoning across modules and about the architecture as a whole.

## Output

Write findings to `ARCHITECTURE_REVIEW.md` at repo root. **Do not read it until Phase 3** — reading
prior findings first anchors the review on yesterday's agenda. See Phase 3.

---

## Phase 0 — Map and inventory (no findings yet)

Build a mental model and an inventory of existing fitness functions. Write nothing to the report.

```bash
# Module + package graph
git ls-files '*.scala' | sed 's|/[^/]*\.scala$||' | sort -u
git ls-files '*.cpp' '*.cu' '*.h' | sed 's|/[^/]*$||' | sort -u

# JNI surface (both sides) and the backend seam
grep -rn '@native' --include='*.scala' .
grep -rn 'extern "C"\|JNIEXPORT' --include='*.cpp' --include='*.h' .

# Inter-module dependency direction (does menger-common stay pure?)
grep -rn 'import menger' --include='*.scala' menger-common/ | grep -v 'import menger.common' || true
grep -rn 'import.*optix\|import.*badlogic' --include='*.scala' menger-common/ || true

# Inventory existing fitness functions — CREDIT these, do not re-nominate
grep -rln 'com.tngtech.archunit' --include='*.scala' .          # ArchUnit rules
git ls-files | grep -iE 'archit|benchmark|integration-tests|manual-test'
test -f ARCHITECTURE_REVIEW.md && echo "prior review exists (read in Phase 3)" || true
# coverage ratchet, version-consistency, memory-leak gates: see .git_hooks/pre-push
test -f .git_hooks/pre-push && echo "pre-push gate present"
```

Cross-reference the real tree against arc42 (read, don't edit): `docs/arc42/05-building-block-view.md`
(§5), `09-architectural-decisions.md` (§9), `10-quality-requirements.md` (§10), `11-risks-and-technical-debt.md`
(§11). Note drift mentally for each axis.

Write down (scratch, not the report): the module graph, the JNI/backend seam, and the inventory of
existing fitness functions. This is the frame handed to every axis subagent.

---

## Phase 1 — Spawn the four axis subagents (in parallel)

Use the `Task` tool, one subagent per axis. Each gets: its remit below, its arc42 anchor, the
Phase-0 map, the fitness-function inventory, and the disjointness contract above. **No subagent may
"review the codebase"** — each has a narrow remit. Each returns 1–6 grounded findings with module/
file evidence and an enforcement verdict (**guarded / partially guarded / unguarded**).

### Axis 1 — Soundness · anchor: arc42 §5

- Module dependency **direction**: does `menger-common` stay pure (no `optix-jni` / LibGDX / JNI
  types upward)? Does `optix-jni` leak native types into `menger-common` / `menger-app`?
- **Cycles** between modules/packages.
- **Boundary integrity**: does `menger-app` depend only on stable interfaces of the other two?
- **Cohesion / god-objects** (e.g. `InteractiveEngine` accreting render + stats + save + exit).
- **Type-level invariants**: dimensions (3D/4D), color spaces, coordinate frames, GPU handles as
  types vs raw `Double`/`Long`.
- **Trust-boundary input validation**: CLI / scene-file parsed and validated before use.
- **arc42 §5 drift**: documented building-block view vs real module graph.
- **Fitness-function audit**: do `ArchitectureSpec` / `ArchitecturePhase2Spec` actually cover the
  dependency/cycle invariants you find? Name uncovered ones — do not propose rules that exist.

### Axis 2 — Maturity · anchor: arc42 §10, §11

- **Error model**: one representation across the seam, or many ad-hoc ones? Strategy vs instances.
- **Test-architecture *shape*** (not %): unit / integration / visual-regression / memory-leak /
  determinism. Gaps in *kind* (e.g. JNI-boundary fault injection, long-running scenarios).
- **Correctness reproducibility**: canonical scenes, caustics analytic→statistical→image ladder,
  render determinism — are correctness claims governed?
- **Observability**: logging coherence; `--stats-json` / `benchmark.sh` as signals.
- **Tech-debt aging**: `CODE_IMPROVEMENTS.md`, arc42 §11 — tracked and shrinking, or accreting?
- **Governance**: credit existing gates (coverage ratchet ≥80%, 4-file version consistency,
  Valgrind / compute-sanitizer, pre-push hook).
- **arc42 §10/§11 drift**: stated scenarios/risks vs reality.

### Axis 3 — Evolvability · anchor: arc42 §9

- **Change-cost**, in files/modules touched, of representative changes:
  add a geometry type (cf. fitness function `M4: < 1 day`); add/extend a dimension (3D→4D —
  parameter or duplicated paths?); add a material/shader; swap or add a render backend.
- **Backend-portability seam**: OptiX hard-wired or abstracted (cf.
  `docs/architecture/optix-jni-decoupling.md`)? Evolvability, not portability-as-goal.
- **LibGDX lock-in** of windowing/UI.
- **Open/closed**: do new features plug into seams or force edits to stable core?
- **Known smell to quantify**: a new rendering feature must currently be added to
  `integration-tests.sh` **and** `manual-test.sh` **and** the shader path — measure this
  shotgun-surgery cost; nominate a parity fitness function if unguarded.
- **arc42 §9 drift**: are recorded ADRs still honored?

### Axis 4 — Performance Architecture · anchor: arc42 §10.4

Performance **as a structural property** — **not** raw benchmarking (`benchmark.sh` owns that).

- Does an abstraction layer sit in the **hot path** (per-ray / per-pixel / inner kernel) and tax it?
- Does the module split force JNI-seam crossings in tight loops, or let the hot path stay hot?
- Algorithmic-complexity decisions visible in structure (surface subdivision `O(12^n)` vs
  `O(20^n)`) — structurally locked in or easy to regress?
- **Is the performance budget a fitness function?** Credit `PerfCheck` + §10.4 budgets +
  `benchmark.sh` baselines; flag which budgets are *advisory* vs *enforced*; nominate promotion
  where a regression would be expensive.
- GPU tunables (block size, stream count, batch size) changeable without recompile?
- **arc42 §10.4 drift**: documented budgets vs measured baselines.

---

## Phase 2 — Synthesize

Collect findings from all four axes. Apply in order:

1. **Drop anything `/code-review` or a linter owns** (see disjointness contract).
2. **Drop anything not grounded** in module/file evidence. "Possible coupling in the renderer" is
   not a finding; "`menger-app/.../InteractiveEngine.scala` owns render+stats+save+exit, 4
   responsibilities, 600+ lines" is.
3. **Merge same-pattern findings** into one.
4. **Rank by architectural blast radius**, not line count.

Target **5–15 findings**. Fewer than 3 with no caveat means you under-looked; more than 15 means you
are reporting linter/seam items or not merging.

---

## Phase 3 — Reconcile with prior review

**Now**, not before, read `ARCHITECTURE_REVIEW.md` if it exists. Reading it earlier anchors on
yesterday's agenda; a finding that survives a no-prior-context pass is a stronger signal.

- **Overlap** old/new → merge, keep new framing, carry old context.
- **Old finding absent from new pass** → verify at its cited location that the pattern is gone (not
  just moved); mark "Resolved" or "Likely resolved — verify."
- **Old finding still real but below the priority cut** → "Carried forward," one line.

---

## Phase 4 — Write `ARCHITECTURE_REVIEW.md`

Renumber findings from 1 by priority each run; `axis + title + location` is the stable identifier.

```markdown
# Architectural Review — YYYY-MM-DD

## Summary
2–4 sentences: biggest structural observation, overall health, largest single risk. No buzzwords.

## Axis scorecard
| Axis | Health | Headline finding |
|------|--------|------------------|
| Soundness          | green/yellow/red | ... |
| Maturity           | ...              | ... |
| Evolvability       | ...              | ... |
| Performance Arch.  | ...              | ... |

## Fitness-function status
| Invariant | Guarded by | Status | Action |
|-----------|-----------|--------|--------|
| no menger-common → optix-jni dep | ArchitectureSpec | guarded | — |
| perf budget P2 (<500ms)          | PerfCheck (advisory) | partial | nominate: enforce |
| add-geometry script parity       | none | unguarded | nominate: parity test |

## Findings
### 1. <title naming the actual structural problem>
**Axis:** Soundness | Maturity | Evolvability | Performance
**Where:** module(s) / `path:line`
**Impact:** Critical/High/Medium/Low — one sentence why this severity.
**Effort:** hours / days / weeks
**Enforcement:** guarded / partially guarded / unguarded
**What:** concrete, evidenced (quote 1–5 lines if it clarifies).
**Why it matters:** real consequence, not "reduces maintainability."
**Suggested direction:** the direction, not a full plan. If unguarded, the nominated fitness function.

### 2. ...

## arc42 coherence
Per-section drift (§5, §9, §10, §11): doc vs reality, one line each.

## Carried forward from prior review
## Resolved since last review
## Positive patterns worth preserving   (0–3, no padding)
```

### Severity calibration

- **Critical** — causes structural correctness failure, or blocks a known upcoming change.
- **High** — significantly slows every new feature in this area, or makes the system hard to reason about.
- **Medium** — real but localized; becomes High within months if ignored.
- **Low** — real but minor. Don't generate Low findings to fill space.

### What does *not* belong

- Anything `/code-review` or a linter catches (seam correctness, `var`/`null`, line length).
- Generic SOLID/clean-code restatements without a specific instance.
- Speculation ("might become a problem if you ever do X").
- Findings without module/file evidence.
- Long lists of "Large Class" — pick the worst 1–2 if size is genuinely the problem.

### Uncertainty labels (use sparingly)

`(certain)` · `(judgment)` · `(needs verification)`. Default to none.
