# Sprint 28: Agentic Development Guardrails & Release Automation

**Sprint:** 28 - Agentic Development Guardrails & Release Automation
**Status:** 🔄 In Progress (started 2026-06-11)
**Estimate:** ~45 hours
**Branch:** `feature/sprint-28`
**Dependencies:** None. Builds on Sprint 27.0's CI hardening of the standalone
`menger-common` and `optix-jni` repositories (MiMa, coverage ratchet, release gates).
**Feature ID:** F14 in [FEATURE_DEPENDENCIES.md](FEATURE_DEPENDENCIES.md)
**Scope agreed 2026-06-10** after discussion; supersedes the earlier
"Industrial-Strength Release & QA" draft.

---

## Goal

Guardrails for agentic development: every quality policy that matters becomes
**structural** — a hook, a CI gate, a merge check, or a bot — with instruction files
(AGENTS.md etc.) as fallback rather than mechanism. Instruction files are
probabilistic controls on AI behavior; gates are not.

Five driving requirements (user, 2026-06-10):

1. All three repos (menger on GitLab; menger-common, optix-jni on GitHub) follow the
   same quality standards — enforced by drift detection, not discipline.
2. Hooks ensure AI-delivered code meets required standards while staying fast and
   unobtrusive (friction proportional to risk).
3. GitLab/GitHub runners always run on the local development system without
   overloading it.
4. Code and architecture reviews at highest standards by several different AI models
   (bias minimization), kept current.
5. Everything automated so that no process step can be forgotten.

Plus: releases trigger on MR merge (opt-out via `NORELEASE` label — a conscious
decision at merge time), and a release must provably install and run on the user's
system from the package.

---

## Success Criteria

- [ ] Shared standards files (scalafix/wartremover/scalafmt configs, hook scripts,
      check scripts) are byte-identical across all three repos; a scheduled drift
      check fails loudly when they diverge
- [ ] pre-commit hook completes in <5 s; pre-push tiers: docs-only ~10 s,
      Scala-only ≤4 min, native changes full (~8–10 min)
- [ ] A commit that modifies tests without a justification trailer is rejected by
      hook; a rendering-relevant change without reference-image updates in the same
      commit is rejected by hook
- [ ] gitlab-runner and GitHub Actions runner run as systemd services, auto-start on
      boot, restart on failure, and are resource-capped (CPU/memory); GPU jobs are
      serialized; a heartbeat alert fires if no runner picks up jobs
- [ ] Every MR/PR gets reviews from ≥2 different AI model families (Claude +
      DeepSeek), posted as structured comments with disagreements preserved
- [ ] Merging an MR without `NORELEASE` produces a published release whose package
      installs and renders on a clean image of the user's OS, automatically
- [ ] The enforcement-audit table maps every AGENTS.md/TESTING.md/RENDERING.md
      policy to its enforcement mechanism; unenforced policies have open issues
- [ ] All tests pass

---

## Tasks

### Task 28.1: Cross-Repo Quality Standards + Drift Detection

**Estimate:** 6h
**Status:** 🔄 In Progress (started 2026-06-11)

Progress:
- [x] `standards/` canonical directory (README, manifest, scalafix.conf); root
      `.scalafix.conf` aligned byte-identical (was missing trailing newline)
- [x] `scripts/check-standards-drift.sh` (`--local`/`--remote`), drift path
      negative-tested
- [x] `scripts/sync-standards.sh` (worktree-safe sibling resolution); run against
      local checkouts — found real drift in optix-jni (comments stripped) and a
      one-byte drift in menger-common; both synced, uncommitted, awaiting review
- [x] CI jobs `StandardsParity` (every pipeline) and `StandardsDrift` (scheduled)
- [ ] Commit + push the synced `.scalafix.conf` in menger-common and optix-jni
- [ ] Create the GitLab pipeline schedule — **deferred to end of sprint** (user
      decision 2026-06-11): set up together with 28.3's runner resource limits and
      the nightly deep tier, after those exist
- [ ] Document the canonical rule in the sibling repos' CONTRIBUTING/README

GitLab `include:` and GitHub reusable workflows are incompatible, so the portable
layer is **scripts and config files, not CI YAML**.

**Implementation:**
- `standards/` directory in the menger repo as the canonical source: scalafix,
  wartremover, scalafmt configs; hook scripts; check scripts (version consistency,
  coverage ratchet invocation, policy checks from 28.2). Each repo's CI is a thin
  platform wrapper invoking these scripts
- Sync mechanism: `scripts/sync-standards.sh` pushes the canonical files to
  menger-common and optix-jni working copies (manual run, reviewed like any change
  in those repos — no silent cross-repo writes)
- Drift detector: scheduled GitLab job fetches `standards/` from both GitHub repos
  via raw URLs, compares hashes against canonical, fails + opens an issue on
  divergence. Read-only GitHub token as masked CI variable
- Document the "menger is canonical" rule in each repo's CONTRIBUTING/README

---

### Task 28.2: Tiered Hooks with Agentic Policy Checks

**Estimate:** 8h
**Status:** 🔄 In Progress (started 2026-06-11)

Progress:
- [x] New fast pre-commit (0.11 s measured): branch guard, staged hygiene
      (conflict markers, debug println, >5 MB files), version consistency
      (extracted to `scripts/check-version-consistency.sh`, shared with
      pre-push), local standards parity. No compile, no tests.
- [x] Shared policy checks in `standards/hooks/` (parity-tracked in manifest):
      `check-test-justification.sh` (modified/deleted `src/test/` files need a
      `Test-Change:` trailer; added tests exempt) and
      `check-rendering-discipline.sh` (pushes touching
      `standards/rendering-paths.txt` patterns need reference-image updates in
      the push or `No-Render-Impact:` trailers). Both verified positive and
      negative against real history.
- [x] pre-push Phase 0 runs the policy checks per push range before anything
      expensive; graceful skip on branches predating standards/hooks.
- [ ] Wire shared policy checks into menger-common/optix-jni hooks (PRs there)
- [ ] ccache for the CUDA side (cheap native rebuilds)
- [ ] Scala-only pre-push timing tier (requires deciding whether integration
      tests may be skipped for non-rendering Scala changes — discuss)

**Finding (for 28.6/28.9):** `docs/TESTING.md` and `docs/RENDERING.md` are
referenced by AGENTS.md but do not exist — the policies the hooks now enforce
lack their authoritative documents. Write them in 28.9.

Note: hooks are wired via `.git/hooks` symlinks into the **main checkout's**
`.git_hooks/`, so the new hooks take effect when this branch reaches the main
checkout; worktree pushes until then use the old hooks.

Friction proportional to risk; policies enforced, not trusted.

**Implementation:**
- **pre-commit (<5 s, new):** scalafmt/scalafix-lint on staged files only, merge
  markers, debug-print patterns, accidental large binaries, version consistency.
  Never compiles
- **pre-push tiers (extend existing change-aware skip):**
  - docs-only diff → validation only (~10 s; exists, formalize)
  - Scala-only → unit tests + scalafix + coverage, skip native rebuild/Valgrind
    (≤4 min; add ccache for the CUDA side to make the boundary cheap to cross)
  - native/shader changes → full hook unchanged (~8–10 min, authoritative)
- **Agentic policy checks (the new substance):**
  - Test files modified → commit message must carry a `Test-Change:` justification
    trailer (enforces the TESTING.md protocol mechanically)
  - Rendering-relevant paths changed (shaders, materials, scene builders — path
    list maintained in `standards/`) → same commit (or stated follow-up commit ref)
    must touch reference images or carry a `No-Render-Impact:` trailer
  - Reject commits authored on `main`
- Hook scripts live in `standards/` (28.1) so all three repos enforce identically

---

### Task 28.3: Local Runner Hardening

**Estimate:** 5h
**Status:** 🔄 In Progress (started 2026-06-11)

Progress:
- [x] Survey: gitlab-runner (system service, Restart=always, enabled) and one
      GitHub Actions runner (optix-jni; enabled, **no Restart=**) both exist.
      Machine: 20 cores / 30 GB.
- [x] Found misconfiguration: the GPU runner entry used `concurrent = 1`,
      which is not a valid per-runner key (correct: `limit = 1`) — GPU jobs
      were never actually serialized (global `concurrent = 2`)
- [x] Design correction vs. plan: GitLab jobs run in **docker containers**, so
      a systemd slice cannot cap them — job caps go through `cpus`/`memory` in
      the runner's docker config; the slice caps the GitHub runner + its jobs
      (child processes) and the runner daemons themselves
- [x] `infra/ci-runners/`: ci-runners.slice (CPUQuota 1000 %, MemoryHigh 10 G,
      MemoryMax 12 G, IOWeight 50), service drop-ins, idempotent pkexec
      install script, README (architecture, limitations, update policy)
- [x] `RunnerHeartbeat` CI job (schedule-gated; alive when the deferred
      nightly schedule is created)
- [x] Apply config.toml fix (`limit = 1`, container cpus/memory; backup at
      config.toml.bak-sprint28) — applied + verified 2026-06-11
- [x] Install slice + drop-ins, daemon-reload — applied + verified 2026-06-11
      (`systemctl show` confirms Slice= and Restart= on both services)
- [ ] Restart services in an idle window (end of sprint, with schedule
      creation); verify runtime slice membership via `systemd-cgls`
- [ ] GitHub-side heartbeat (with sibling hook-wiring PRs, 28.2 follow-up)

Accepted limitation (documented): GitLab and GitHub GPU jobs can still run
simultaneously (independent queues, one GPU); revisit with a GPU lock if it
bites.

**Implementation:**
- gitlab-runner **and** GitHub Actions self-hosted runner (for menger-common /
  optix-jni GPU jobs) as systemd services: start on boot, `Restart=on-failure`
- Both services in one systemd slice: `CPUQuota` ~50 % of cores, `MemoryMax` (e.g.
  16G), `IOWeight` below interactive use — CI must never make the desktop unusable
- GPU contention is structural: one GPU shared with interactive work → GPU-tagged
  jobs serialized (`concurrent = 1` per runner, GPU tag required); document that
  queueing is the accepted trade-off
- Heartbeat: scheduled pipeline job on each platform; if not picked up within N
  minutes, a notification fires (GitLab pipeline-failure email / ntfy webhook)
- Runner update policy documented (pinned versions, monthly bump via 28.8 MRs)

---

### Task 28.4: Multi-Model Review Automation

**Estimate:** 10h
**Status:** 🔄 In Progress (started 2026-06-11)

Progress:
- [x] `standards/review-guidelines.md` — versioned code review guidelines: what
      to flag (alpha convention, JNI boundary, agentic anti-patterns, arch
      violations) and what NOT to flag (scalafix/WartRemover already handles)
- [x] `standards/architecture-review-guidelines.md` — arc42-conformance layer:
      module boundary table, JNI contract rules, build/CI change checks
- [x] `standards/architecture-paths.txt` — paths that trigger architecture
      review (build defs, JNI native, module public APIs, CI YAML, arc42 docs)
- [x] `scripts/review.sh` — platform-agnostic core: Claude (claude-sonnet-4-6)
      + DeepSeek (deepseek-chat) via respective APIs → consolidated findings
      JSON with agreed/single-model labels; 4000-line diff truncation guard
- [x] `scripts/adapters/gitlab-mr-review.sh` — idempotent bot note on MR
      (create or update using `<!-- menger-ai-review-v1 -->` marker)
- [x] `scripts/adapters/github-pr-review.sh` — idempotent bot review on PR
      (COMMENT event, dismisses prior review before posting fresh)
- [x] `AIReview` CI job in `review` stage (new stage between preconditions and
      test): runs on every MR, `skip-review` label skips it,
      `allow_failure: true` (advisory until confidence built), auto-detects
      architecture-relevant diffs, posts findings to MR
- [x] arc42 §11: added TR-9 (DeepSeek data flow risk) and TR-10 (GPU
      contention risk from 28.3)
- [x] Guidelines added to `standards/manifest.txt` for cross-repo parity
      tracking
- [ ] Wire GitHub sibling repos to use the adapter (with 28.2 hook-wiring PRs)
- [ ] Promote `allow_failure: false` once a sprint of reviews shows low
      false-positive rate (conscious decision at sprint 29 close)
- [ ] Add DEEPSEEK_API_KEY as masked+protected CI variable (user action
      required: Settings → CI/CD → Variables)

≥2 model families per review to minimize single-model bias; disagreements are
signal, not noise.

**Implementation:**
- Platform-agnostic core: `review.sh <diff> <guidelines>` → findings JSON. Thin
  adapters post to GitLab MR discussions and GitHub PR reviews
- Reviewers: Claude (claude-code CLI non-interactive or API) + **DeepSeek** (user
  decision 2026-06-10). DeepSeek key as masked, protected CI variable; never
  logged; note in arc42 §11 that review diffs are sent to the DeepSeek API
  (acceptable for this public codebase — revisit if private code appears)
- Consolidation step: dedupe findings by file/line/category; findings flagged
  `agreed` (both models) vs `single-model` — single-model findings are *kept*,
  labeled, because disagreement is the bias-detection mechanism
- Review guidelines versioned in-repo (`standards/review-guidelines.md`,
  `standards/architecture-review-guidelines.md`); reviews always read the current
  version; sprint-close skill gains a "refresh guidelines" step (28.9)
- Architecture-review trigger: MRs touching architecture-relevant paths (JNI
  boundary, shader pipeline, module structure, build definitions) additionally run
  an arc42-conformance review (model reads relevant arc42 sections + diff)
- Default on for every MR/PR in all three repos; `skip-review` label as the
  conscious-decision escape hatch
- Out of scope: auto-applying review findings — humans (or the authoring agent in
  a later session) act on comments

---

### Task 28.5: Release-on-Merge + Installable-Package Proof

**Estimate:** 8h
**Status:** 🔄 In Progress (started 2026-06-11)

Progress:
- [x] `scripts/release.sh --prepare --version X.Y.Z` — bumps all four version
      files (build.sbt, .gitlab-ci.yml, MengerCLIOptions.scala,
      docs/guide/user-guide.md + docs/USER_GUIDE.md) and inserts CHANGELOG
      stub. Never infers version — user provides it. `--check` delegates to
      `check-version-consistency.sh`.
- [x] `NORELEASE` label support — `TagIsNewAndConsistent` and
      `ChangelogIsUpdated` now also skip on `$CI_MERGE_REQUEST_LABELS =~
      /NORELEASE/`; `CreateRelease` checks both `#norelease` in title AND
      `NORELEASE` in MR labels via the same commits API call.
- [x] `Test:InstallSmoke` CI job in `release` stage — runs on tag pipeline,
      needs `BuildDeployable` artifact, unzips the package on the local nvidia
      runner, runs `xvfb-run menger-app --objects type=cube-sponge:level=1
      --timeout 0.1` (headless smoke render proving the installed package runs
      on the user's GPU).
- [ ] Wire `Test:InstallSmoke` as a required gate before `PushToGithub` (add
      to `PushToGithub.needs` once a release has run smoke-green at least once)
- [ ] Add `Test:Debian`/`Test:Ubuntu` automatic rules for tag pipelines
      (deferred: smoke + existing manual jobs sufficient until external users
      materialize — revisit per BACKLOG_FEATURES.md)

A merge **is** the release decision unless labeled otherwise.

**Implementation:**
- GitLab: pipeline rule on merge to main checks MR labels via API; `NORELEASE`
  label → skip release stage. Otherwise: validate the MR carried a version bump
  (all four version files consistent and != last tag — versions are never
  inferred, the MR author sets them; `scripts/release.sh --prepare` automates the
  bump mechanics), then tag, build, publish artifacts, update CHANGELOG anchor
- Installable-package proof: the existing manual `Test:Debian`/`Test:Ubuntu` jobs
  become **automatic in the release stage**, pinned to include the user's Ubuntu
  version; plus one GPU smoke render of the *installed* package on the local
  runner (install → render sphere → health-check passes)
- GitHub repos: tag-publication flow from Sprint 27.0 already exists; align
  trigger semantics (release-on-merge + `NORELEASE`) so all three repos behave
  identically
- Rollback note: a bad release is followed by a fix-forward release; document

---

### Task 28.6: Enforcement Audit — Policy → Mechanism Map

**Estimate:** 3h
**Status:** ✅ Done (2026-06-11)

**Deliverables:**
- `docs/ENFORCEMENT.md`: all policies from AGENTS.md and the release checklist
  mapped to their enforcing mechanism; 8 ❌ gaps identified
- GitLab issues #155–#162 opened for every unenforced row
- GitLab merge checks configured: *Pipelines must succeed* + *All discussions
  must be resolved* (confirmed via API — both now true)

**Implementation:**
- Table in `docs/ENFORCEMENT.md`: every policy from AGENTS.md, TESTING.md,
  RENDERING.md, and the release checklist → enforcing mechanism (pre-commit /
  pre-push tier / MR gate / merge check / scheduled job / **unenforced**)
- Every `unenforced` row gets a GitLab issue (implement or consciously accept)
- Configure GitLab merge checks: pipeline must succeed, discussions resolved —
  the merge button is the single funnel
- This document is the living definition of the quality system; reviewed at
  sprint close

---

### Task 28.7: Performance-Regression Guard (slim)

**Estimate:** 4h

"The AI made it slower" is an agentic failure mode like any other.

**Implementation:**
- Benchmark scene set (sphere, sponge-volume L4, menger4d L3, IBL+accumulation),
  `--stats-json` output (small CLI addition)
- Compare on pre-merge pipeline + nightly against rolling baseline (median of last
  5 green runs); >15 % ms/frame regression fails. Medians of 3 runs; relative
  thresholds (GPU-runner variance)

---

### Task 28.8: Dependency-Update Automation

**Estimate:** 3h

- sbt-updates is broken on sbt 1.12.6 (TODO.md): use Renovate (native GitHub app
  for the two GitHub repos; self-hosted/CI-scheduled for GitLab) or Scala Steward —
  spike, pick one, document in arc42 §9; weekly bump MRs/PRs in all three repos
- Native deps (CUDA, OptiX SDK, libav) excluded — driver/SDK matching stays manual
  (TROUBLESHOOTING.md)

---

### Task 28.9: Documentation + Process Wiring

**Estimate:** 3h
**Status:** ✅ Done (2026-06-11)

- arc42 §10: quality scenarios for the guardrail system; §11: risks (DeepSeek data
  flow, single-GPU queueing, runner availability)
- TESTING.md: flaky-test policy — known flakes (sponge-volume,
  tesseract-with-material) get one auto-retry, max 2 sprints on the retry list
  before mandatory root-causing
- sprint-close skill: add "refresh review guidelines" + "review ENFORCEMENT.md"
  steps
- CHANGELOG.md entry

---

## Gates Placement (agreed 2026-06-10)

Every MR is fully gated **before merge**; pushes during development stay fast:

| Tier | When | What |
|------|------|------|
| Fast | every push to an MR branch | change-aware: unit + scalafix + coverage always; sanitizers when native changed; integration suite when rendering-relevant paths changed |
| Full | pre-merge (pipeline for merged results) | full integration suite, Valgrind, compute-sanitizer subset, perf compare (28.7) |
| Deep | nightly | everything unconditionally: full sanitizers, full perf set, drift check (28.1), heartbeat |

Rationale: literal everything-on-every-push would serialize ~30–40 min pipelines on
a single local GPU and throttle agentic throughput; the merge decision still sees
the complete picture.

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 28.1 | Cross-repo standards + drift detection | 6h |
| 28.2 | Tiered hooks + agentic policy checks | 8h |
| 28.3 | Local runner hardening (GitLab + GitHub, systemd, slices, GPU serial) | 5h |
| 28.4 | Multi-model review automation (Claude + DeepSeek, arch trigger) | 10h |
| 28.5 | Release-on-merge + installable-package proof | 8h |
| 28.6 | Enforcement audit + merge checks | 3h |
| 28.7 | Performance-regression guard (slim) | 4h |
| 28.8 | Dependency-update automation | 3h |
| 28.9 | Documentation + process wiring | 3h |
| **Total** | | **~50h nominal; ~45h expected via 28.1/28.2 shared scripts and 28.5 reuse of existing Debian jobs** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green (using its own new tiered form)
- [ ] CHANGELOG.md updated
- [ ] One release has shipped end-to-end through the merge-triggered pipeline,
      including the automatic install-and-render proof
- [ ] Drift check and heartbeat have each fired at least once in anger (test by
      intentionally diverging a standards file / stopping a runner)

---

## Explicitly Out of Scope (decisions 2026-06-10)

- Security/supply-chain axis: SBOM, CVE scanning, provenance (deferred)
- Auto-applied review findings
- Multi-GPU / environment-matrix testing (documented matrix only)
- .deb/Flatpak end-user packaging beyond the existing package + install proof
  (revisit when external users materialize)
