# Investigation: fractional 4D sponge corner artifact during 4D rotation

**Date:** 2026-06-03
**Reporter:** user (interactive viewing of Sprint 26 task 26.7 GPU-only 4D path)

## Prior report (secondary — to be verified)

Fractional tesseract sponges with **1 < level < 2** show "occasional weird
artifacts when 4D rotating", "always near a corner of the original cube".
Levels ≤ 1 look fine. (Cause-neutral re-render + agreed description pending —
Rule 1 checkpoint.)

Context: task 26.7 made GPU 4D projection the only path. A separate
`gpu4DPlan` fix (apply `quadsBuffer` skin offset) removed an all-over
z-fighting noise on fractional sponges; this corner artifact is a residual,
rotation-dependent defect on the 1<level<2 fractional pair.

## Stage 0 — reproduce

**Command:** rotation sweep, fixed stage binary:
`menger-app --headless -s /tmp/rot_$a.png --plane y:-2 --objects type=tesseract-sponge:level=1.5:size=0.8:rot-xw=$a:rot-yw=20` for a ∈ {0,20,35,45,55,70,90}.

**Output/measurement:**
- rot-xw=0, 90: clean (no artifacts).
- rot-xw=70 (and other oblique angles): thin dark rectangular dashes/slivers
  appear on the surface, distinct from the cubic sponge holes, concentrated
  near the projected cube's edges/corners. PNG size grows with angle
  (17.8K@0 → 20.9K@90), i.e. more high-frequency pixel variation at oblique
  angles.

**Description v1 (cause-neutral, pending user agreement):**
On fractional tesseract sponges (1 < level < 2), at oblique 4D rotation
angles, thin dark rectangular sliver/dash marks appear on the surface —
separate from the cubic holes — clustered near the cube's edges/corners.
Not present at level ≤ 1, nor at rot-xw 0/90.

**Hypothesis update:** consistent with thin geometry seen near edge-on
appearing only at oblique projection angles on the fractional pair.
Cause NOT yet investigated (Rule 1 checkpoint first).

## Stage 1 — detector (validated)

`SkinOffsetGapSpec` (menger-app test). Replicates `quadsBuffer`'s exact
per-face offset `(Σ f.normals) * offset` and counts source-coincident vertex
groups that diverge after the offset (gap-opening).

Fixture validation:
- offset = 0 → 0 gaps (no movement). PASS.
- two coplanar faces, identical winding, shared edge → 0 gaps. PASS (known-good).
- two perpendicular faces sharing an edge → gap detected. PASS (known-bad).

## Stage 2 — localization

| sponge level | faces | skin-offset gaps |
|---|---|---|
| 0 (cube skin) | 24 | 16 |
| 1 (sponge skin) | 1152 | 256 |
| 2 | 55296 | 8192 |

**Root-cause mechanism (confirmed by reading `Face4D.normals`):** the offset
direction is `Σ f.normals`, where `normals` are **winding-dependent SIGNED**
unit normals (`Face4D.scala:103` `normalSigns(e.take(2))`). So each quad face
is displaced rigidly along a per-face diagonal that depends on its winding and
can point inward. Faces sharing a vertex but with different windings move that
vertex in different directions → it splits → a gap that exposes the level-(n+1)
surface behind as an extra dark square. The defect is only visible at level ≥ 1
because the holes introduce interior faces (whose windings/offsets diverge most)
clustered at the cube's corners; the level-0 cube skin's 16 gaps are on exterior
edges over a coincident level-1 surface and are not visible (detector
over-reports visibility — counts correlate with, but are not equal to, the
visible defect; final check is by eye after the fix).

**Prime suspect:** `Mesh4DGpuFlatten.quadsBuffer` per-face signed-normal offset.

**Proposed fix:** replace the per-face normal offset with a gap-free uniform
radial expansion about the sponge centre (origin): `v -> v * (1 + offset)`.
Source-coincident vertices scale identically → 0 gaps by construction, while
the level-n skin still sits just outside the level-(n+1) surface (no z-fight).

## Stage 3 — fix + verification

**Fix:** `Mesh4DGpuFlatten.quadsBuffer` — replaced the per-face signed-normal
offset with a uniform radial scale about the origin-centred sponge:
`v -> v * (1 + skinOffset)`. Source-coincident vertices scale identically →
gap-free by construction. (`gpu4DPlan` already routes the fractional lower-level
mesh through `quadsBuffer` with the skin offset; that wiring is unchanged.)

**Detector (regression test) result:** `SkinOffsetGapSpec` now drives the real
`quadsBuffer` and asserts 0 gaps at sponge levels 0/1/2 (was 16/256/8192).
Self-validation fixtures (coincident→0, divergent→>0) pass.

**Visual confirmation (primary criterion):** re-rendered the rotation sweep
(level 1.5, rot-xw=70 — the worst angle) and the full fractional set
interactively; the extra dark squares are gone and no z-fight noise returned.
User confirmed clean under 4D rotation (2026-06-04).

**Status: RESOLVED.** Regression test: `SkinOffsetGapSpec`.

## Description v-final (agreed with user, 2026-06-03)

On fractional tesseract sponges (1 < level < 2), at oblique 4D-rotation
angles, **extra dark square marks** appear on the surface, distinct from the
intended cubic sponge holes, clustered near the cube's edges/corners. Absent
at level ≤ 1 and at rot-xw 0°/90°. (User: "squares", agreed equivalent to my
"dashes/slivers" seen at oblique angle.)

**Refinement:** level 0.5 is also a fractional pair (skin = level-0 solid
cube) and renders clean; the defect needs the skin level ≥ 1 (a sponge with
many faces/corners). → severity scales with skin-level corner count → points
at the per-face skin-offset geometry (`Mesh4DGpuFlatten.quadsBuffer`, added
in the z-fight fix), not the projection kernel itself.
