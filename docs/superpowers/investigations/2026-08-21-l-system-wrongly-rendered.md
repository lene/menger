# Investigation: wrongly rendered L-System (manual-test #13)

Three scenes: `fern3d` L3 (169), `hilbert3d` L4 (170), `tree` 4D L3 (172).

## Stage 0 — Context, render and describe — 2026-08-21

**Commands:**
```
./menger-app/target/universal/stage/bin/menger-app --headless \
  --objects type=lsystem:preset=fern3d:level=3:size=0.8 --plane y:-2 -s /tmp/fern3d.png
./menger-app/target/universal/stage/bin/menger-app --headless \
  --objects type=lsystem:preset=hilbert3d:level=4:size=1.2 --plane y:-2 -s /tmp/hilbert3d.png
./menger-app/target/universal/stage/bin/menger-app --headless \
  --objects "type=lsystem:preset=tree:level=3:size=1.5:dim=4:rot-xw=15:rot-yw=10:eye-w=3:screen-w=1.5" \
  --plane y:-2 -s /tmp/tree4dl3.png
```

**Description (agreed with user, cause-neutral):**
- fern3d L3: a single vertical stack of ~12 ring/disc-shaped segments directly on top of each
  other — no branching, no lateral spread. Not fern-shaped.
- hilbert3d L4: one short, smoothly-curved tube with only ~3 bends — far too few for a
  level-4 space-filling curve.
- tree (4D) L3: a handful of small disconnected fragments floating above/behind the camera,
  no visible trunk or branch structure, disconnected from the ground.

All three look severely under-generated relative to their preset+level — shared cause in the
L-system generation path, not three unrelated defects.

## Stage 0 — Codebase map

`LSystemPresets.scala` (grammar/angle/segment defaults) → `LSystemGrammar.rewrite` (string
rewriting per iteration) → `LSystemSceneBuilder.generateFromSpec` (constructs
`LSystemTurtle3D`/`LSystemTurtle4D` from the rewritten string, dispatches on `dim`) →
turtle's `generate()`/`process()`/`stepSymbol` (interprets the string as movement commands,
accumulating points into "runs" flushed to `ObjectSpec(objectType="curve")` at branch
boundaries) → `CurveSceneBuilder.buildScene` (uploads each run as one OptiX curve primitive).

## Root cause 1 — silently dropped single-F branches

Simulated the grammar expansion directly (Python, matching `LSystemGrammar`'s exact
rewrite rule) to rule out an expansion-level bug first:

**Command/measurement:**
```python
rewrite("F", {'F': "F[&F]F[^F][&F]"}, 3)  # fern3d L3
# -> 404 chars, 125 F's, 93 brackets
rewrite("X", {'X': "^<XF^<XFX-F^>>XFX&F+>>XFX-F>X->", 'F': "F"}, 4)  # hilbert3d L4
# -> 17551 chars, 4095 F's
```
**Hypothesis update:** expansion is correct and non-trivial (thousands of F's at level 4).
The bug is entirely in turtle interpretation, not generation.

Read `LSystemTurtle3D`/`LSystemTurtle4D` in full. `emitRun` silently drops any run with fewer
than 2 points (`if points.length < 2 then (specs, (Vector.empty, Vector.empty))`). Every site
that starts a fresh run (`generate()`'s initial call, `stepFwdNoRecord`, `stepPop`) reset the
run to an *empty* vector rather than seeding it with the turtle's current position — so a
branch containing exactly one `F` (the overwhelmingly common case in these grammars, e.g.
`[&F]`) produced a 1-point run that got silently discarded, and trunk continuity broke across
every `]` for the same reason.

Traced by hand through fern3d's rule `F[&F]F[^F][&F]` from a fresh (empty-seeded) turtle:
the very first trunk segment (origin→p1) is never its own point (origin is never recorded);
the two brackets separated by a trunk `F` survive as 2-point curves; the *third* bracket
(`][&F]`, immediately adjacent to the previous `]`, no intervening trunk `F`) produces a
1-point run and is dropped entirely. Confirms the observed "sparse, disconnected, mostly
missing" symptom.

**Fix:** seed every fresh run with `Vector(state.pos)`/`Vector(state.width)` (or the
`popped`/new state's position after `]`) instead of `Vector.empty`, at all four reset sites
in each turtle (`generate()`, `stepFwdNoRecord`, `stepPop` ×2 branches, plus `stepF`'s
sphere-shape case in `LSystemTurtle3D` for consistency, though none of these three presets
exercise it).

## Root cause 2 — LSystemTurtle3D's `<`/`>` roll uses the wrong axis

hilbert3d has zero `[`/`]` in its grammar, so root cause 1 does not touch it (its whole
expansion is one continuous run, flushed once at the end). Read `stepSymbol`'s command table
directly: `<`/`>` rotated around `state.up` — the *same* axis as `+`/`-` (yaw) — instead of
`state.heading` (the correct roll axis; `\`/`/` already used `state.heading` correctly).
`LSystemTurtle4D`'s `<`/`>` were already correct (`headingIdx`/`anaIdx`, a genuinely distinct
plane from `+`/`-`'s `headingIdx`/`leftIdx`).

hilbert3d's rule string is dominated by `<`/`>` (far more than `+`/`-`/`&`/`^` combined).
Aliasing roll onto yaw collapses most of the curve's intended 3D turning into the same 2D
yaw plane, explaining the too-simple render.

**Fix:** `<` and `>` now rotate around `state.heading`, matching `\`/`/`'s existing (correct)
roll-axis convention.

## Checkpoint (prime suspect) — user confirmed both causes; proceed with both fixes

## Fix implementation and test fallout — 2026-08-21

Implemented both fixes in `LSystemTurtle3D.scala` / `LSystemTurtle4D.scala`. Ran the existing
`LSystemTurtle3DSuite`/`LSystemTurtle4DSuite`/`LSystemSceneBuilderSuite` (51 tests): 4 failed,
all due to the seed point shifting array indices by exactly one (a width/point-count
assertion written against the old, buggy "no seed point" behavior):
- `"FFF"` now has 4 points (12 floats), not 3 (9) — the seed anchors the run.
- Two width-decay tests compared `widths(1)` vs `widths(0)*0.5`; with the seed now occupying
  index 0, the correct comparison is `widths(2)` vs `widths(1)*0.5` (3D) /
  `widths(1..4)` instead of `widths(0..3)` (4D, `F!F!F!F`).

Investigated each failure individually before touching any assertion (per `docs/TESTING.md`):
all four are the expected, correct consequence of the fix, not new defects. Updated the
four assertions with an inline comment explaining the index shift; added a
`Test-Change:` trailer to the fix commit.

**New regression tests** (directly assert the fixed behavior, not just index-shifted old
behavior):
- `LSystemTurtle3DSuite`/`LSystemTurtle4DSuite`, "Single-F branches... not be silently
  dropped": `"F[+F][-F]"` (two adjacent single-F branches, no intervening trunk `F`) —
  `specs.length shouldBe 2`. Would have been `1` before the fix (second branch dropped).
- `LSystemTurtle3DSuite`, "Roll (< and >)... rotate around the heading axis": `"F<F"` — the
  two segments must stay collinear (a roll around one's own heading is a no-op on heading
  direction). Would have failed before the fix (roll aliased to yaw, breaking collinearity).

## Visual verification — 2026-08-21

Rebuilt, re-rendered all three scenes headless:
- **fern3d L3**: default (head-on) camera still shows a mostly-vertical solid tube (branches
  from `&`/`^` tilt in the Y-Z plane, foreshortened from a head-on Z-facing camera) — but a
  3/4 camera angle (`--camera-pos 1.5,0.5,1.5 --camera-lookat 0,0.3,0`) shows a genuine
  branching fractal frond shape. Instance count log confirms 75 curve instances generated
  (auto-adjusted max-instances 64→150), consistent with real branching, not a handful of
  surviving fragments.
- **hilbert3d L4**: improved (a connected loop instead of scattered rings) but still does not
  look like a proper space-filling curve from either camera angle. See "Root cause 3" below
  — determined to be a separate, unrelated rendering-primitive issue, not a turtle bug.
- **tree (4D) L3**: improved (a connected chain of segments instead of disconnected
  fragments) but still floats away from the ground plane. Filed as a residual (see below) —
  likely a 4D-projection artifact from running a 3D-only preset (no `<`/`>`) through the 4D
  turtle, a pre-existing open question from before this investigation, not addressed here.

## Root cause 3 (investigated, not fixed) — hilbert3d's B-spline smoothing

User asked to keep investigating hilbert3d rather than file it immediately. Added a
throwaway test printing `LSystemTurtle3D.HilbertCurve3D.generate()`'s point count and
bounding box (removed after use, never committed):

**Measurement:** 2801 points, `x ∈ [-0.253, 0.253]`, `y ∈ [-0.253, 0.253]`,
`z ∈ [-0.5, 0.5]` (post-`normalize()`), first 10 points show genuine grid-aligned
right-angle jumps consistent with a real Hilbert-curve traversal of a cube's edges.
**Hypothesis update:** the turtle-generated data is correct and non-degenerate. The
remaining defect, if any, is downstream of the turtle.

Read `CurveSceneBuilder.buildScene`: passes all points/widths straight to
`renderer.addCurveInstance(...)`, no truncation or subsampling in Scala. Grepped optix-jni's
native curve setup: `curve_input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE`
(`OptiXContext.cpp:812`, consistently declared across `OptiXContext.cpp`/`PipelineManager.cpp`/
`OptiXApiBindings.cpp`). A cubic B-spline is a smooth curve influenced by, but not passing
through, its control points — exactly the kind of primitive that would round off thousands of
closely-spaced sharp corners into a smooth path, matching the observed spiral.

**Conclusion:** not a bug. `CurveSceneBuilder`/`addCurveInstance` correctly render exactly
what they are asked to render; the mismatch is that every L-system "curve" run, regardless of
turn sharpness, is rendered through the same smooth B-spline primitive. That's the right
choice for organic shapes (fern/tree branches, where smoothing is invisible or desirable) and
the wrong one for a geometric space-filling curve with thousands of sharp turns in one
continuous run. A real fix (corner-sharpening via duplicate-point insertion at sharp turns)
is scoped feature work affecting every L-system preset's rendering — **filed as a residual,
not implemented in this investigation** (user decision).

## Minimum artefacts checklist

- [x] Regression tests exist (`LSystemTurtle3DSuite`/`LSystemTurtle4DSuite`, permanent) for
      both fixed root causes
- [x] Fix commit passes those tests (and the 4 pre-existing tests updated for the correct,
      shifted indices)
- [x] This note records per-step commands, measurements, root-cause narrative for all three
      root causes investigated (2 fixed, 1 filed as residual)
- [x] `ManualTestNeedFixing.md` #13 corrected: marked PARTIALLY RESOLVED, prior "needs
      deciding" framing replaced with the actual two confirmed bugs, and the two residuals
      (hilbert3d B-spline smoothing, tree/dim=4 positioning) filed explicitly rather than
      left unstated
- [x] No `CODE_IMPROVEMENTS.md` entry existed for this
- [x] Residuals filed with rationale, not silently dropped
