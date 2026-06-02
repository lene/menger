# Remove Legacy CPU 4D Projection Path (Sprint 26 — Task 26.7)

> **For agentic workers:** REQUIRED SUB-SKILL — use `superpowers:executing-plans` (or
> `superpowers:subagent-driven-development`) to run this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. This touches rendering output: the integration
> suite (`scripts/integration-tests.sh`) MUST pass on the final commit, and any
> reference-image diffs resolved in the same commit.

## Goal

Retire the CPU 4D→3D projection path and make GPU 4D projection always-on. Today
`MeshFactory.createUpload` chooses between a GPU plan (`Mesh4DGpuFlatten`) when
`gpuProject4D = true` and a CPU plan (`Mesh4DProjection`) otherwise. The `gpuProject4D`
flag — configurable through `RenderConfig` / the DSL and `false` by default in several
builders — selects a user-visible rendering mode. This task removes that mode.

## Why this is NOT "delete dead code"

- `Mesh4D`, `Mesh4DProjection`, `Mesh4DGpuFlatten` are all **active**.
- `Mesh4DProjection` is the CPU projector ("Renders any 4D mesh by projecting it to 3D");
  it is also (mis)used by the GPU path purely as a container to read `.mesh4D`.
- `RotatedProjection` is already gone (no class; only a stray test-name string).
- There is a documented interactive hang workaround (`H-mixed-frac-int-interactive-hang`)
  that exists specifically because of the `gpuProject4D = off` path.

So this is a deliberate **rendering-behavior change** + API change, requiring reference
re-verification.

## Blast radius

| Action | File | Change |
|---|---|---|
| Modify | `menger-common/.../RenderConfig.scala` | remove `gpuProject4D` field (line 20) |
| Modify | `menger-app/.../dsl/RenderSettings.scala` | remove `gpuProject4D` DSL field (25) + `.getOrElse` wiring (44) |
| Modify | `menger-app/.../engines/scene/MeshFactory.scala` | `createUpload` drops `gpuProject4D` param; always GPU path for `isProjected4D`; `gpu4DPlan` must always succeed (turn `None` into an error) or extract `mesh4D` directly without `Mesh4DProjection` |
| Modify | `menger-app/.../engines/scene/TriangleMeshSceneBuilder.scala` | remove `gpuProject4D` ctor param + the `if gpuProject4D` branches (78, 85, 160, 162, 238) |
| Modify | `menger-app/.../engines/GeometryRegistry.scala` | remove `gpuProject4D` param (37, 52) |
| Modify | `menger-app/.../engines/BaseEngine.scala` | drop `renderConfig.gpuProject4D` args (~10 sites) |
| Modify | `menger-app/.../engines/InteractiveEngine.scala` | remove `gpuProject4D` branches (218, 457-472, …); resolve the hang-workaround branch now that the CPU path is gone |
| Modify | `menger-app/.../engines/WithAnimation.scala` | remove `gpuProject4D` branches (126, 147, 157, 161) |
| Modify | `menger-app/.../objects/higher_d/TesseractMesh.scala`, `TesseractSpongeMesh.scala`, `TesseractSponge2Mesh.scala` | change return type from `Mesh4DProjection` to the raw `Mesh4D` (or a thin 4D-mesh result) consumed by `Mesh4DGpuFlatten` |
| Delete | `menger-app/.../objects/higher_d/Mesh4DProjection.scala` | once nothing constructs it |
| Delete | `menger-app/.../test/.../higher_d/Mesh4DProjectionSpec.scala` | CPU-projection unit test |
| Modify | tests referencing `gpuProject4D` / `Mesh4DProjection` | `Project4DGpuSuite`, `FractionalLevelSceneBuilderSuite`, polytope mesh suites, etc. |
| Modify | `docs/...` , `CODE_IMPROVEMENTS`/sprint | note CPU path removed; close task 26.7 |

## Approach

`Mesh4DProjection` currently carries `(mesh4D, eyeW, screenW, rot*)` and can both
CPU-project and expose `.mesh4D`. The GPU path only needs `mesh4D` + the
`Projection4DSpec`. So:

1. Introduce (or reuse) a lightweight carrier for `mesh4D` (the `Mesh4D` itself plus the
   `Projection4DSpec` already on `ObjectSpec`). `MeshFactory.mesh4DProjection` becomes
   `mesh4DFor(spec): Option[Mesh4D]`.
2. `gpu4DPlan` builds the flat buffer directly from that `Mesh4D` via
   `Mesh4DGpuFlatten.facesBuffer`.
3. `createUpload` always produces `Gpu4D` for `isProjected4D` specs; a `None` (unknown 4D
   type) becomes a hard error rather than a silent CPU fallback.
4. Delete `Mesh4DProjection` + spec once unreferenced.

## Tasks

- [ ] **Task 1 — Make GPU projection unconditional in MeshFactory (TDD).**
  Write/adjust a `MeshFactorySuite` test asserting every `isProjected4D` spec yields a
  `Gpu4D` plan (no `Cpu` fallback). Refactor `mesh4DProjection` → `mesh4DFor: Option[Mesh4D]`,
  `gpu4DPlan` to flatten from `Mesh4D`, and `createUpload` to drop the `gpuProject4D`
  param. Update `TesseractMesh`/`TesseractSpongeMesh`/`TesseractSponge2Mesh` return types.

- [ ] **Task 2 — Remove `gpuProject4D` from config + DSL.**
  Delete the field from `RenderConfig` and `RenderSettings`; fix `SceneConverter` wiring.
  Update DSL tests/examples.

- [ ] **Task 3 — Remove `gpuProject4D` from the builder/engine layer.**
  `TriangleMeshSceneBuilder`, `GeometryRegistry`, `BaseEngine`, `InteractiveEngine`,
  `WithAnimation`. Collapse the now-dead branches. Re-evaluate and remove the
  `H-mixed-frac-int-interactive-hang` workaround (its triggering path no longer exists);
  keep a regression test for the interactive fractional path.

- [ ] **Task 4 — Delete `Mesh4DProjection` + `Mesh4DProjectionSpec`.**
  Confirm zero references, delete, recompile.

- [ ] **Task 5 — Update remaining tests.**
  `Project4DGpuSuite`, `FractionalLevelSceneBuilderSuite`, polytope mesh suites — drop
  `gpuProject4D`, assert GPU-only behavior.

- [ ] **Task 6 — Rendering verification.**
  Run `PARALLEL_MODE=false ./scripts/integration-tests.sh ./menger-app-$VERSION/bin/menger-app`.
  Resolve any 4D reference-image diffs in the same commit (regenerate only after manual
  confirmation that GPU output is correct). Verify both `scripts/integration-tests.sh`
  and `scripts/manual-test.sh` still exercise every 4D shape.

- [ ] **Task 7 — Docs + close-out.**
  Update arc42 if the module/quality picture changes; note the CPU path removal; mark
  task 26.7 done in the sprint file.

## Risks

- 4D reference images may shift if CPU vs GPU projection differed sub-pixel. Inspect
  diffs; do not blindly regenerate.
- The interactive hang workaround interaction (`H-mixed-frac-int-interactive-hang`) must
  be re-validated, not just deleted.
- `Mesh4DGpuFlatten` must support every shape the CPU path did (tesseract, sponges,
  pentachoron, 16/24/600/120-cell). Verify each renders via the GPU path before deleting
  the CPU fallback.
