# Investigation: `tree`/`dim=4` renders as an oversized blob, not rooted on the ground plane

Residual from H3.6 (`ManualTestNeedFixing.md` #13), scenes 172/173: `preset=tree` (a 3D-only
L-system grammar with no `>`/`<` ana/kata symbols) routed through the 4D turtle (`dim=4`).
H3.6 had already fixed the single-F-branch and roll-axis bugs shared by every preset; this
residual was specific to the 3D-preset-through-4D-turtle combination and was left open with
the description "renders floating away from the ground plane... possibly a 4D-projection
artifact."

## Checkpoint — description agreed — 2026-08-23

User confirmed treating this as a real bug to investigate and fix, not a product question
about whether the combination should be allowed at all.

## Stage 1 — first hypothesis (wrong) — rotation-induced projection distortion

**Hypothesis:** `LSystemTurtle4D` applies a default `rotXW=15°/rotYW=10°` rotation to every
point regardless of grammar content. For `tree` (which never leaves `w=0`), this mixes the
local X/Y extent into a nonzero `w`, and `Projection.apply`'s position-dependent factor
`(eyeW-screenW)/(eyeW-w)` would then distort points differently depending on their distance
from the root — read as "leaning away from vertical."

**Fix attempted:** in `LSystemSceneBuilder`, detect whether the rewritten grammar string
contains `>`/`<`; if not, suppress the default rotation (use 0°/0° instead of 15°/10°)
unless the user explicitly overrode it.

**Verification — this hypothesis was wrong.** Rendered before/after:
`--objects type=lsystem:preset=tree:dim=4:level=3 --plane y:-2`. The "fixed" render was
visually indistinguishable from the original — still two giant blocky segments where the 3D
render shows dozens of fine branches. Measured horizontal centroid drift top-vs-bottom of
the rendered silhouette: <2px in a 400px-wide frame for *all three* variants (3D, 4D before,
4D after) — the original "leaning" read was a misjudgment of the tree's own natural
asymmetric branching shape, not a real horizontal drift. **Reverted the rotation-suppression
change** rather than ship a fix that doesn't fix anything.

## Checkpoint — reported the failed hypothesis, asked how to proceed — 2026-08-23

User: continue investigating now.

## Stage 2 — corrected symptom description

Cropped and visually compared the base/root region of both renders. The 3D tree shows dozens
of small branch segments forming a recognizable fractal frond. The 4D-routed tree shows only
**two** large, blocky, angular shapes — not a leaning tree, a structurally different (much
coarser, apparently merged) render.

## Stage 2 — detector — 2026-08-23

Pure-Scala comparison (no GPU needed): generated the identical rewritten grammar string for
`tree` at level 3 through both `LSystemTurtle3D` and `LSystemTurtle4D`, counted emitted
`ObjectSpec`s and their per-curve point counts.

**Result:** both turtles emit exactly **63** curve specs with **identical** point-count
breakdowns per curve. Segmentation/branch logic is byte-for-byte equivalent between the two
turtles — ruled out any run-merging or branch-count bug.

## Stage 3 — root cause — 2026-08-23

Dumped actual coordinate values for corresponding curves. 4D coordinates were consistently
several times larger in magnitude than the corresponding 3D coordinates for the same grammar
and segment length. Isolated with a minimal `"FFF"` grammar (segment length 1, no turns, no
rotation):

- 3D output: Y range `[-0.5, 0.5]` (4 points, evenly spaced at 1/3 apart)
- 4D output (rotXW=rotYW=0): Y range `[0.0, 1.5]` (evenly spaced at 1/2 apart — the raw,
  unscaled segment length)

**Found it:** `LSystemTurtle3D` has a `normalizeScale: Boolean = true` constructor parameter.
`generate()` calls a private `normalize()` step that computes the bounding box of *all*
generated points, rescales everything so the largest dimension is exactly 1.0, and centers
the result on the origin. `LSystemTurtle4D` has **no equivalent parameter or step at all** —
`generate()` returns the raw `process(...)` output directly.

For a multi-level fractal, the raw (un-normalized) accumulated turtle-walk distance is much
larger than what normalization would produce — for `tree` at level 3, several times larger.
Combined with `tree`'s root sitting at local origin and growing asymmetrically (never
centered), the un-normalized 4D output ends up drastically oversized relative to the scene's
expected framing, and — critically — individual branch cylinders (still at the small
`initWidth` radius) become so long relative to their radius, and so densely packed at that
scale, that they visually merge into a small number of blob-like solid shapes instead of a
recognizable sparse branching structure. This fully explains both the "floating"
misdescription (the object's apparent extent bears no relation to where a normalized,
similarly-placed 3D object would sit) and the "wrongly rendered" structural collapse.

## Checkpoint — fix proposed and confirmed — 2026-08-23

Proposed extracting `LSystemTurtle3D`'s `normalize`/`normalizeSpec` into a shared,
dimension-agnostic utility (`LSystemNormalization`, operating on already-3D-projected
`ObjectSpec`/`CurveData` points) and wiring `LSystemTurtle4D.generate()` to call it, matching
the 3D turtle's `normalizeScale = true` default. User confirmed.

## Stage 3 — fix — 2026-08-23

- New `menger.engines.scene.LSystemNormalization` object: `normalize(List[ObjectSpec])`,
  extracted verbatim from `LSystemTurtle3D`'s former private methods.
- `LSystemTurtle3D.generate()` now delegates to `LSystemNormalization.normalize`; its own
  copies of `normalize`/`normalizeSpec` deleted.
- `LSystemTurtle4D` gained a `normalizeScale: Boolean = true` constructor parameter;
  `generate()` now calls `LSystemNormalization.normalize` when enabled, matching the 3D
  turtle's contract exactly.

**Verification:**
- Minimal-grammar unit check: 4D `"FFF"` output now normalizes to the same `[-0.5, 0.5]`
  range as 3D's, confirming the shared utility works identically for both.
- Full test suite: all 2491 existing tests pass (154 suites), including all pre-existing
  `LSystemTurtle3DSuite`/`LSystemTurtle4DSuite`/`LSystemSceneBuilderSuite` cases (54 tests)
  unchanged.
- New regression tests in `LSystemTurtle4DSuite`: `generate` normalizes 4D output to match
  3D's scale for an identical grammar; `normalizeScale = false` leaves output at raw scale
  (contract parity with `LSystemTurtle3D`).
- Visual crosscheck: `--objects type=lsystem:preset=tree:dim=4:level=3 --plane y:-2` now
  renders fine branching structure correctly rooted on the ground plane, matching the 3D
  version's recognizable shape.
- Checked for regressions on the genuinely-4D presets (`tree4d`, `hilbert4d`, which never
  had normalization either but are not reported defects): both still render reasonably;
  `hilbert4d`'s blobby appearance is `hilbert3d`'s own separate, already-filed corner-
  sharpening defect (H3.6 residual), not something this fix introduced — confirmed by
  rendering `hilbert3d` (3D, already-normalized) and observing the same blobby-not-sharp
  character.
- `integration-tests.sh`'s two `lsystem4d` scenes (`preset=tree` at level 3 and 4 — the
  exact reported defect) had their reference images regenerated; new references verified
  deterministic across three independent renders (0-pixel diff each) before committing.

## Minimum artefacts checklist

- [x] Regression test exists (`LSystemTurtle4DSuite`, 2 new cases, permanent)
- [x] Fix commit passes the full test suite (2491/2491)
- [x] This note records per-step commands, measurements, root-cause narrative, and the
      failed first hypothesis (not hidden — the rotation-suppression attempt and why it was
      wrong)
- [x] Prior incorrect description ("possibly a 4D-projection artifact... floating") corrected
      in `ManualTestNeedFixing.md` #13
- [x] Scripted integration coverage's reference images regenerated and verified deterministic
- [x] No residual defects found for this specific issue (the hilbert corner-sharpening
      residual is separately tracked, unaffected by this fix)
