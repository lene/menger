---
name: debugging-rendering-bugs
description: Use when a rendered image is visually wrong, a render produces uniform/blank output, or geometry has visual defects (missing faces, bright patches, flaps, dark solid where structure expected). Use before any shader, material, or geometry fix attempt.
---

# Debugging Rendering Bugs

## Two Hard Rules — Non-Negotiable

**Rule 1 — Agree on the description with a human before investigating.**
Render the defect yourself. Describe it in cause-neutral language. Iterate until both sides use the same words. A prior report can be wrong about the symptom AND smuggle in cause theories; trusting it sends the investigation into the wrong subsystem.

**Rule 2 — Document every step as you go.**
After each step: exact command run, output/measurement, hypothesis update. Multi-session bugs cost dearly if you re-run experiments already done.

**These rules apply especially under time pressure.**

### Description iteration example

- **v0 (prior report):** "solid dark cube" — wrong symptom; cause theories embedded.
- **v1 (first render):** "triangles with wrong-facing normals" — closer but still a cause claim.
- **v2 (user correction):** "faces only have a neighbour on one edge" — observation-only.
- **v3 (agreed):** "mesh is not closed; some triangles are flaps connected along one edge."

Do not start investigating until the description sounds like v3.

---

## Mandatory User Checkpoints

Pause and confirm with user before continuing past each gate:

1. **After description v-final** — show the rendered defect; agree on wording.
2. **After detector fixture validation** — show results; confirm detector is trustworthy.
3. **After Stage 2 localises the bug** — show evidence table; agree on prime suspect.
4. **After visual crosscheck** — show annotated render; confirm same artifacts as eye saw.

**Never skip a checkpoint because "the evidence is unambiguous."** That is the rationalization that causes wrong-invariant chasing.

**These are hard stops. Do not proceed past a checkpoint without an explicit user response in the current conversation. A prior message from the user does not count — ask again at the checkpoint.**

---

## Stopping Criteria

**Primary: the rendered image matches expected. Secondary: numeric invariant passes.**

When they disagree — **trust the eye, not the number.**

If the image looks correct but the detector still reports failures: stop, file a separate low-priority ticket for residual cases, do **not** keep chasing. See Traps → "Wrong invariant."

---

## The Four Stages

### Stage 0 — Context

- Reproduce the defect yourself with a minimal CLI. Write it into an investigation note.
- Find a parameterization (level, camera angle, scene) that makes the defect unmistakable. Don't debug marginal cases.
- Map the codebase: which files are candidate stages of the pipeline, which existing tests cover them, and what those tests *don't* check.

### Stage 1 — Build a detector

Convert the visual symptom into a **numeric invariant** that a test can assert. Examples:
- Closed-manifold: every mesh edge is shared by exactly two triangles
- Normal consistency: adjacent faces' normals agree on shared edge orientation
- Vertex uniqueness at seams: geometrically-equal vertices have equal float values
- Pixel diff against a known-good reference image

**Validate the detector on known-good AND known-bad fixtures before trusting it.** An unvalidated detector is itself a bug candidate. If the detector has a bug, all Stage 2 data is invalid. Skipping fixture validation is a named anti-pattern.

### Stage 2 — Localize

Apply the detector at each stage of the pipeline (generation → projection → tessellation → merge) and at several difficulty levels. Record a table of results per stage per level. Look for transitions — the level or stage where the invariant first breaks is the prime suspect.

Compare against a **sibling implementation** if one exists. Features unique to the broken variant become prime suspects.

**Always complete the visual crosscheck:** re-render the scene with detector-flagged elements coloured distinctly. Confirm they are the same artifacts the eye saw. Skipping this is a named anti-pattern. "47 bad edges" may be a completely different set from the visible patches.

### Stage 3 — Root cause and fix

Propose the narrowest possible cause compatible with Stage 2's evidence. Implement the smallest fix that targets it. Re-run the detector; re-render; compare to the original defect image.

Make the detector a **permanent regression test** — it already exists at this point.

Rewrite any stale bug report. Replace the prior description with v-final wording (or delete if fully resolved).

**Remove the resolved entry from `CODE_IMPROVEMENTS.md` entirely.** The file header states: "Resolved items are removed from this file entirely — git history is the record of what was fixed." Do not mark it with strikethrough or add a RESOLVED annotation — delete the whole entry.

---

## Red Flags — STOP and Follow the Process

If you catch yourself thinking any of the following, stop and return to the rules:

- "The symptom is clear" — you haven't rendered it yourself
- "I'll trust the prior bug report" — prior reports can be wrong about both symptom and cause
- "We already agreed on the description / no need to re-render" — descriptions need re-verification each session; render it yourself
- "The evidence is unambiguous, I can skip the checkpoint" — this is how wrong-invariant chasing starts
- "I already know the root cause" — you haven't built a detector yet
- "The image looks right but the detector still fires" — trust the eye; file a separate ticket
- "I'll document later" — documentation must happen *as you go*

---

## Common Rationalizations

| Rationalization | Reality |
|-----------------|---------|
| "The symptom is clear from the bug report" | Prior reports are secondary sources; render it yourself |
| "This is a shader/material bug" | Most surprising rendering bugs live upstream in geometry; start there |
| "I can skip fixture validation, the detector is obviously correct" | An unvalidated detector is a bug candidate; Stage 2 data is worthless without validation |
| "Evidence is unambiguous, no need for visual crosscheck" | The numeric property may not match the visible defect; crosscheck closes this loop |
| "Image looks right but there are still 1920 bad edges — must fix them" | Trust the eye over the number; file a separate ticket |
| "I'll document the steps at the end" | Sessions end, context compacts; document after every step or re-run everything |

---

## Traps Specific to Rendering Bugs

**Choosing the wrong invariant.** The numeric property may be correlated with but not equivalent to the visible defect. After the fix, if the image looks right but the detector still fires — stop and check. You may be chasing a different (possibly pre-existing) bug, not the original one.

**Multiple bugs hiding in the same symptom.** Rendering defects often overlap. Resist fixing all of them in one commit — separate detectors mean separable bugs. Defer residuals explicitly.

**Debugging the shader before the geometry.** Shader and material are tempting because their outputs are visual. Most surprising rendering bugs live upstream in geometry generation or topology. Start upstream; swing to the shader only if geometry checks out.

---

## Minimum Artefacts

An investigation is complete when:
- A regression test exists (the detector, as a permanent test)
- The fix commit passes that test
- The investigation note records per-stage commands, measurements, images, and root-cause narrative
- Any prior incorrect description has been corrected or removed
- The entry in `CODE_IMPROVEMENTS.md` has been **deleted** (not struck through — see file header)
- Residual defects discovered during the investigation have their own low-priority tickets

---

## Investigation Note Format

File: `docs/superpowers/investigations/YYYY-MM-DD-<slug>.md`

Per-step template:
```
## <Stage name> — <date>

**Command:** <exact command run>
**Output/measurement:** <what it returned>
**Hypothesis update:** <what this confirms, rules out, or changes>
```

See `docs/superpowers/investigations/2026-04-21-tesseract-sponge-2-flap.md` for a full worked example.

---

## Menger Codebase Notes

**Headless render CLI:**
```
./menger-app/target/universal/stage/bin/menger-app -o --headless \
  -s /tmp/debug.png --objects type=<type>:level=<N>
```
Add `--allow-uniform-render` to bypass the render-health check when testing blank/uniform scenes (exit code 2 otherwise).

**Topology checker (geometry bugs):** `MeshTopology` in test sources —
`checkFace4D(faces)` and `checkTriangleMesh(mesh)`. Returns `isManifold`,
`edgeUseHistogram`, `boundaryFaces`. **Validate** with:
- Unit cube (12 triangles) → should be manifold
- Unit cube minus one triangle → should be non-manifold

See `MeshTopologySpec` for ready-made fixture tests.

**Render health check (blank/uniform output):** `RenderHealth.check` in
`optix-jni/src/main/scala/menger/optix/RenderHealth.scala`. Logs exact
uniform colour and CLI args; exit code 2 = failed render. Check the log
for the offending invocation.

**Preserved diagnostics:** `TopologyDiagnosticSpec` in test sources contains
`ignore`d per-stage topology runs for TesseractSponge2. Flip `ignore` → `it`
to reactivate for new investigations.
