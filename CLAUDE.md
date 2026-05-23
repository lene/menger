# Context

Please strictly follow the project rules and standards defined in @AGENTS.md.

## Branching

- **Never commit directly to `main`.** Always work on a feature branch (e.g.
  `feature/sprint-NN`, `feature/<topic>`). When starting a task, check
  `git branch --show-current`; if it reports `main`, switch to the active
  feature branch (or create one) before making any code or doc changes.
- The active feature branch may live in a worktree under `.worktrees/`.
  Use that worktree rather than the main checkout when iterating.
- `main` is updated only by merging a feature branch (via PR/MR), not by
  direct commits.

## After Every Task — Mandatory DoD Check

**After every task, run `./.git_hooks/pre-push` without being asked.** The pre-push hook is the 
single authoritative DoD gate — it runs unit tests, scalafix, packaging, integration tests, and all
other checks. Do not run checks individually as a substitute; the hook is the source of truth and 
changes over time.

Binary path (needed for manual renders / reference image generation only) - replace X, Y and Z with 
the current version:
- Stage dir: `menger-app-X.Y.Z/` (workspace root)
- Binary: `./menger-app-X.Y.Z/bin/menger-app`
- Headless: `__GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a ./menger-app-X.Y.Z/bin/menger-app --headless ...`

## Long-Running Commands

Always tee output to a file when running long commands (builds, test runs, render suites):

```bash
some-long-command 2>&1 | tee /tmp/output.log
```

Examine `/tmp/output.log` afterward if needed — avoids re-running expensive commands to inspect failures.
