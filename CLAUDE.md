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
