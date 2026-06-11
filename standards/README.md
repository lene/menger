# Shared Quality Standards (canonical source)

This directory is the **canonical source** for quality-standards files shared by the
three Menger repositories:

| Repo | Platform |
|------|----------|
| menger (this repo) | GitLab — canonical |
| menger-common | GitHub (`github.com/lene/menger-common`) |
| optix-jni | GitHub (`github.com/lene/optix-jni`) |

Introduced in Sprint 28 (task 28.1) as part of the agentic-development guardrails:
standard parity across repos is enforced by automated drift detection, not by
discipline.

## How it works

- `manifest.txt` lists every parity-tracked file: `<path-in-repo> <file-here>`.
  Each listed repo path must be **byte-identical** to its canonical file in this
  directory — in all three repos.
- `scripts/check-standards-drift.sh --local` verifies this repo's own copies match
  the canon. Runs in every CI pipeline (`StandardsParity` job) — cheap, seconds.
- `scripts/check-standards-drift.sh --remote` fetches the tracked files from the two
  GitHub repos and compares them against the canon. Runs on a CI schedule
  (`StandardsDrift` job) and fails loudly on divergence.
- `scripts/sync-standards.sh` copies the canonical files into local checkouts of the
  sibling repos. It never commits — changes in the sibling repos are reviewed and
  pushed like any other change there.

## Rules

1. **Edit the canonical file here first**, then run `scripts/sync-standards.sh`, then
   commit in each sibling repo. Never edit a tracked file only in one repo.
2. Repo-specific behavior does not belong in tracked files. If a repo needs an
   exception, the file is not a shared standard — remove it from the manifest
   instead of letting it drift.
3. New shared standards (hook policy scripts, review guidelines — Sprint 28.2/28.4)
   are added by placing the file here and listing it in `manifest.txt`.

## Currently tracked

- `.scalafix.conf` ← `scalafix.conf`
- `standards/hooks/*.sh` — shared hook policy checks (Sprint 28.2): branch guard,
  staged hygiene, test-change justification, rendering discipline. Per-repo hooks
  in `.git_hooks/` stay thin wrappers calling these.

## Repo-specific (in standards/, not parity-tracked)

- `rendering-paths.txt` — grep patterns defining rendering-relevant paths for the
  rendering-discipline check; each repo maintains its own (the check skips cleanly
  when absent).

## Not yet tracked (known gaps)

- WartRemover options live in each repo's `build.sbt` (build definitions are
  repo-specific); parity is by convention until extracted into a shared plugin.
- Wiring the shared policy checks into the sibling repos' hooks happens via
  follow-up PRs there (tracked in SPRINT28.md task 28.2).
