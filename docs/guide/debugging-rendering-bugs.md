# Debugging Rendering Bugs: A Method

Rendering bugs are perceptual. They resist the usual "read the stack trace"
reflex because there is no stack trace — just an image that looks wrong. The
temptation is to guess: normals, lighting, winding, shader. Guessing on a
rendering bug is how days disappear.

This document is a reusable method for investigating hard-to-reproduce
rendering bugs. It was written as the deliverable of the
`TesseractSponge2` "flap" investigation (see
`docs/superpowers/investigations/2026-04-21-tesseract-sponge-2-flap.md`), and
is framed generally so the next rendering bug can follow the same shape.

## Two hard rules

**Rule 1 — Agree on the bug description with a human before investigating.**
Written bug reports about rendering are secondary sources. Render the defect
yourself, describe what you see in cause-neutral language, and iterate with
your collaborator until both sides use the same words. A one-line prior
report can be wrong about the symptom *and* smuggle in false cause theories;
trusting it sends the investigation into the wrong subsystem.

**Rule 2 — Document each step and its result as you go.** Investigations of
hard rendering bugs tend to span multiple sessions. The single biggest time
sink is re-running experiments someone (possibly you, before compaction)
already ran. Append to an investigation note after every step: exact
commands, file paths, measurements, images, hypotheses confirmed or
rejected.

Everything below is guidance. Everything above is non-negotiable.

## A worked example of iterating the description

From the `TesseractSponge2` flap bug:

- **v0 (prior report):** *"solid dark cube at all integer levels"* — wrong
  symptom; smuggled in cause theories (shadow occlusion, lighting angle,
  degenerate normals). Trusting this would have sent us into the shader.
- **v1 (first render):** *"some triangles have outward-facing normals on
  what should be inward-facing surfaces"* — closer to the symptom, but still
  a cause claim about normal direction.
- **v2 (user correction):** *"all faces should be connected at the edges;
  the wrong-looking faces only have a neighbour on one edge"* —
  observation-only; cause-neutral; directly testable.
- **v3 (agreed):** *"the mesh is not a closed 2-manifold; some triangles
  are flaps connected to the rest of the surface along only one edge."*

Each iteration stripped a layer of cause speculation and moved toward a
property measurable without running the shader. Do not start Stage 1 until
the description sounds like v3.

## The stages

### Stage 0 — Context

- Re-reproduce the defect yourself with a minimal CLI. Write that CLI into
  the investigation note.
- Find a parameterization (level, camera, scene) that makes the defect
  *unmistakable*. Don't debug marginal cases.
- Map the codebase: which files are candidate stages of the pipeline, which
  existing tests cover them, and what those tests *don't* check.

### Stage 1 — Build a detector

Convert the visual symptom into a **numeric invariant** that a test can
assert. Examples:

- closed-manifold: every mesh edge is shared by exactly two triangles
- normal consistency: adjacent faces' normals agree on the shared edge's
  orientation
- vertex uniqueness at seams: geometrically-equal vertices have equal float
  values
- bit-identical output under a scene symmetry
- pixel diff against a known-good reference image

Build the detector. Then **validate it on known-good and known-bad
fixtures** before trusting its report on the real bug — otherwise the
detector is just another bug candidate. For the flap investigation the
fixtures were a unit cube (manifold) and a unit cube minus one triangle
(three boundary edges).

### Stage 2 — Localize

Apply the detector at each stage of the pipeline (generation, projection,
tessellation, merge) and at several difficulty levels. Turn "bug somewhere
between input and pixels" into "bug between stage K and stage K+1".

Record a table of results per stage per level. Look for transitions —
the level or stage where the invariant first breaks is the prime suspect.

Also compare against a **sibling implementation** if one exists. If a
near-relative passes the detector, the features unique to the broken
variant become prime suspects.

Close the loop with a **render-and-highlight crosscheck**: re-render the
scene with detector-flagged elements coloured distinctly. Confirm they are
the same artifacts the eye saw. This prevents the disaster of debugging the
wrong numeric property — easy to spot "47 bad edges" and never notice they
are a different set from the visible patches.

### Stage 3 — Root cause and fix

Propose the narrowest possible cause compatible with Stage 2's evidence.
Implement the smallest fix that targets it. Re-run the detector; re-render;
compare to the original defect image.

Make the detector a **permanent regression test** — free at this point, it
already exists — so this defect cannot silently return.

Finally, **rewrite the stale bug report**. If `CODE_IMPROVEMENTS.md` or a
ticket contained the v0 description, replace it with the corrected one
(or delete it if fully resolved). The next reader deserves a correct record.

## Traps specific to rendering bugs

**Choosing the wrong invariant.** The numeric property you chose in Stage 1
may be *correlated with* the visible defect but not *equivalent to* it. In
the flap investigation, "closed 2-manifold" was the obvious invariant but
turned out to be stricter than the actual visible objective ("no wrong-
facing triangles"): the fix eliminated visible flaps while leaving the mesh
technically non-manifold. If you find yourself chasing a residual count of
detector failures after the rendered image looks right, stop and check
whether you are still working on the original bug or on a new, separate
one. **Render the image; trust the eye over the number.**

**Multiple bugs hiding in the same symptom.** Rendering defects often
overlap. The flap investigation surfaced three interleaved issues: a
normal-orientation bug (fixed), a float-precision seam bug (fixed as a
side-effect), and a sub-cube containment bug (deferred as a separate
ticket). Resist the urge to fix all of them in one commit — separate
detectors mean separable bugs.

**Debugging the shader before the geometry.** Shader, lighting, and camera
are tempting because their outputs are visual. But most surprising
rendering bugs live upstream, in geometry generation or topology. Start
upstream and work downstream — you can always swing back to the shader if
the geometry checks out.

## Minimum artefacts to leave behind

A rendering-bug investigation is considered complete when:

- A failing-test reproducer exists (the detector as a regression test).
- The fix commit passes that test.
- The investigation note under
  `docs/superpowers/investigations/YYYY-MM-DD-<short-slug>.md` records the
  per-stage commands, measurements, images, and root-cause narrative.
- Any prior incorrect description (in `CODE_IMPROVEMENTS.md`, ticket, or
  docs) has been corrected or removed.
- Residual defects discovered during the investigation have their own
  tickets, scoped tightly.

If any of these is missing, the next person to look at this code area pays
the cost.
