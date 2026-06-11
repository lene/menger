# Local CI Runner Hardening (Sprint 28.3)

Both CI runners (GitLab for menger, GitHub Actions for optix-jni/menger-common GPU
jobs) run on the local development machine. This directory makes them **always
available** and **never disruptive**.

## Architecture

| Concern | GitLab runner | GitHub Actions runner |
|---------|---------------|------------------------|
| Always up | vendor unit: `Restart=always`, enabled | drop-in adds `Restart=on-failure` (vendor unit had none), enabled |
| CPU/RAM/IO caps | jobs run in **docker containers** → capped via `cpus`/`memory` in `/etc/gitlab-runner/config.toml` (the systemd slice cannot reach them); the runner process itself joins the slice | jobs are child processes of the service → `ci-runners.slice` caps runner **and** jobs |
| GPU serialization | `limit = 1` on the GPU-tagged runner entry (note: the per-runner key is `limit`; a per-runner `concurrent` key is silently ignored — this was misconfigured before Sprint 28.3) | one registered runner = one job at a time by design |

`ci-runners.slice`: CPUQuota 1000 % (10 of 20 cores), MemoryHigh 10 G / MemoryMax
12 G (of 30 G), IOWeight 50. CI queues rather than starving the desktop.

## Files

- `ci-runners.slice` → `/etc/systemd/system/`
- `gitlab-runner-override.conf` → `/etc/systemd/system/gitlab-runner.service.d/override.conf`
- `github-runner-override.conf` → `/etc/systemd/system/<actions.runner.*>.service.d/override.conf`
- `install-runner-hardening.sh` — installs the above via pkexec, idempotent;
  restart the services manually when no jobs are running

`/etc/gitlab-runner/config.toml` is **not** managed from this repository (it holds
runner tokens). The Sprint 28.3 changes applied there, for reference:

```toml
# global
concurrent = 2            # one menger job + one job from other registrations

# on the gitlab.com / GPU runner entry only:
[[runners]]
  limit = 1               # serializes menger GPU jobs (was: invalid 'concurrent = 1')
  [runners.docker]
    cpus = "10"           # half the cores per job container
    memory = "12g"
```

## Known limitations (accepted)

- **Cross-platform GPU contention:** a GitLab GPU job and a GitHub GPU job can
  still run simultaneously (independent queues, one GPU). Accepted trade-off;
  revisit with a flock-style GPU lock if it bites in practice.
- **Interactive GPU use** is not protected from CI GPU jobs (and vice versa);
  rendering-test flakes under contention are documented in TESTING.md.

## Heartbeat

`RunnerHeartbeat` job in `.gitlab-ci.yml` (schedule-gated): a trivial job that
fails the scheduled pipeline visibly if no runner picks it up — pipeline-failure
notification is the alert. Becomes active when the nightly schedule is created
(deferred to end of Sprint 28 by user decision 2026-06-11). The GitHub-side
equivalent lands with the sibling-repo hook-wiring PRs (28.2 follow-up).

## Update policy

Runner binaries are pinned by installation, not auto-updated. Monthly: check
`gitlab-runner --version` against GitLab's current, and the Actions runner
self-update messages in its service log; update during an idle window, then
re-run `install-runner-hardening.sh` (drop-ins survive package updates, but
verify with `systemctl show gitlab-runner -p Slice`).
