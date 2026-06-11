# Rendering Change Discipline

A *rendering change* is any commit that can affect rendered pixel output.
This document defines what counts, what the discipline requires, and common traps.

---

## What counts as a rendering change

Any modification to these paths is a rendering change:

- `menger-geometry/src/main/native/` — shaders, BVH builders, hit programs
- `menger-app/src/main/scala/menger/engines/scene/` — scene configuration, materials
- `menger-app/src/main/scala/menger/objects/` — geometry generators
- `menger-app/src/main/scala/menger/optix/` — OptiX launch and parameter passing

The canonical list is `standards/rendering-paths.txt` — that file is the source of
truth used by `check-rendering-discipline.sh`.

When in doubt, assume the change is rendering-relevant.

---

## The discipline

**Rule:** If a commit touches rendering-relevant paths, the integration suite must
run on that commit and any reference-image diffs must be committed in the **same
push** (or the immediately following commit on the same branch).

**Do not let the pre-push hook be the discovery mechanism.**
Stale references accumulating across multiple commits make bisection painful and
cause spurious integration-test failures at push time.

---

## Enforcement

The `check-rendering-discipline.sh` hook (pre-push Phase 0) checks every commit
in the push range:

- If any commit touches `standards/rendering-paths.txt` patterns **and** the push
  range does not also touch `scripts/reference-images/`, the push is rejected.
- Exception: any such commit may carry a `No-Render-Impact: <reason>` trailer to
  assert it provably does not change pixel output (e.g. a comment-only edit or a
  refactor with identical codegen).

---

## Local verification: always use sequential mode

Run the integration suite in sequential mode to avoid GPU-contention flakes:

```bash
PARALLEL_MODE=false ./scripts/integration-tests.sh ./menger-app-$VERSION/bin/menger-app
```

Parallel mode (`MAX_PARALLEL_JOBS > 1`) can produce non-deterministic pixel
differences under GPU memory pressure — sequential mode is the authoritative
reference for reference-image updates.

---

## The sequential-vs-parallel trap

The CI pipeline runs jobs in parallel by default. If you update reference images
using parallel mode locally, the sequential-mode integration test that runs on push
may still fail. Always generate reference images with `PARALLEL_MODE=false`.

---

## New rendering features

Every new rendering feature (material preset, object type, shader path, CLI
parameter) must be added to **both**:

1. `scripts/integration-tests.sh` — headless regression scenario
2. `scripts/manual-test.sh` — human visual verification scenario

Both scripts must have at least one test that exercises the new code path before
the feature is considered done. This is not currently enforced by a gate (open
issue [#156](https://gitlab.com/lilacashes/menger/-/work_items/156) and
[#157](https://gitlab.com/lilacashes/menger/-/work_items/157)).

---

## Alpha channel convention

**This is a hard invariant — getting it wrong causes silently incorrect rendering.**

- `alpha = 0.0` → **fully transparent** (no opacity, no absorption)
- `alpha = 1.0` → **fully opaque** (full opacity, maximum absorption)

Applies everywhere: OptiX shaders, Beer-Lambert absorption, Scala `Color`, tests.
