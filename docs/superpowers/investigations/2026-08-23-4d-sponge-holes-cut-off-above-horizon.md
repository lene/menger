# Investigation: 4D sponge small holes cut off above the camera's horizon line

`ManualTestNeedFixing.md` #8: `tesseract-sponge` (levels 2-3) reportedly renders flat/hole-free
above the ground plane's horizon, correct below it. Counterexample noted in the original
report: plain `tesseract` (level 0, no sponge recursion) renders clean everywhere.

## Checkpoint — first conclusion (wrong) — 2026-08-23

**First pass:** built a row-based pixel-sampling detector comparing texture-variance above vs
below the horizon for `--objects type=tesseract-sponge:level=3:size=2.8:rot-y=35`. Concluded
there was no bug — that the appearance was fully explained by sparse fractal holes plus
perspective foreshortening (higher rows sample a different, genuinely flatter face by
coincidence). Presented this conclusion to the user.

**User rejected it**, with two screenshots from the live interactive app and this description:
*"these pictures show a clear difference above and below the horizon. especially on the second
one: you can clearly see the 1st level hole on the right side of the sponge both above and
below the horizon, but the 2nd level, smaller holes clearly are cut off at the horizon line.
below it they are clearly visible, above the horizon not. this horizon clearly continues across
the entire sponge."* The detector's fixed x-column sampling range had a confound: at different
rows it was unknowingly comparing genuinely different faces, not the same face split by the
horizon.

## Checkpoint — corrected description agreed — 2026-08-23

Re-scoped: not "does detail differ above/below" in general, but specifically "does 2nd-level
hole detail on the *same visible face* disappear exactly at the horizon row, independent of
which face happens to be in view."

## Stage 1 — clean reproduction

- Straddling the horizon: `--camera-pos 0,0,4 --camera-lookat 0,0,0 --objects
  type=tesseract-sponge:level=3:size=2.8:rot-y=35 --plane y:-2` — same object, single frame,
  camera level. Below-horizon portion of the object shows small 2nd-level holes; above-horizon
  portion of the *same face* renders flat.
- Camera-only-shift control: `--camera-pos 0,-1.2,4 --camera-lookat 0,-1.2,0` (same object,
  same rotation, only the camera moves so the whole object sits above the horizon) — renders
  completely flat/hole-free everywhere. This rules out per-face visibility differences as the
  explanation; only the ray directions changed.

Ruled out (each tested independently, effect persisted or was absent as noted): antialiasing,
denoising, accumulation-frames, IBL (not configured), shadow-ray max-depth exhaustion
(`--max-ray-depth 1` on the *working* below-horizon view did not break it, disproving a
coverage-alpha depth-exhaustion hypothesis). A sphere (analytic `hit_sphere.cu`) in the same
steep-look-angle configuration showed a real shading gradient, not flatness — pointed at
`hit_triangle.cu`/the mesh path specifically, not a shared/generic rendering bug.

## Checkpoint — shader instrumentation approved — 2026-08-23

## Stage 2 — shader instrumentation

Temporarily overrode `hit_triangle.cu`'s `__closesthit__triangle()` RGB output with diagnostic
values (raw `geom.normal` XYZ mapped to RGB), republished optix-jni locally (had to clear both
`~/.ivy2/local` **and** the coursier cache each iteration — a single `sbt publishLocal` alone
was silently ignored by menger), and rebuilt/re-rendered.

**Result:** the raw (pre-entering-flip) normal was constantly `(0, 1, 0)` — straight up — for
literally every triangle in the mesh, regardless of the true per-face direction. The
`entering` classification (`dot(ray_direction, normal) < 0`) then tracked `ray_direction.y`'s
sign almost perfectly instead of true geometry: for a level camera, screen rows above center
have `ray_direction.y > 0`, rows below have `< 0` — exactly matching the observed horizon-line
split. Position data was unaffected (correct silhouette/holes render, confirming this was a
normal-only defect, not a mesh-generation defect).

## Stage 3 — root cause

Traced the actual render path: `tesseract-sponge` normalizes to `tesseract-sponge-volume`,
which is in `ObjectType.PROJECTED_4D_TYPES`, which routes through `MeshFactory`'s `Gpu4D` plan
→ `renderer.setProjectedMesh` → `OptiXWrapper::setTriangleMesh4DQuads` →
`project4d_faces_kernel` (`optix-jni/src/main/native/project4d.cu`) — **not** the CPU
`Mesh4DProjection.toTriangleMesh` path, which is dormant for this object type today.

`project4d_faces_kernel` computes each face's normal via Newell's method (correct in general)
and falls back to a hardcoded `(0, 1, 0)` when it judges the face degenerate:

```cuda
float n_len2 = nx * nx + ny * ny + nz * nz;
if (n_len2 < 0.0001f) {
    nx = 0.0f; ny = 1.0f; nz = 0.0f;
}
```

Newell's-method magnitude scales with face area (edge²), so `n_len2` scales with edge⁴.
`TesseractSponge` shrinks faces by 1/3 per recursion level. At level 3, faces are
`(1/3)³ ≈ 0.037×` the base tesseract's edge length, so `n_len2 ≈ 0.037⁴ ≈ 1.9×10⁻⁶` — about
50× *below* the fixed `0.0001f` threshold, even though the faces are perfectly
non-degenerate. Every face at that recursion depth trips the "degenerate" branch and gets the
same wrong constant normal — exactly the observed symptom. (Verified empirically that
`TesseractSponge`'s own face generation is sound: compiled and ran the real classes directly,
0/2,654,208 level-3 faces have duplicate/collinear corners or near-zero area under a
*correct* area check.)

The identical flaw exists in the CPU sibling, `Mesh4DProjection.scala`'s
`faceToTriangleMesh` (`if nl < 0.0001f then (0f, 1f, 0f)`) — dormant for `tesseract-sponge`
specifically, but live for any future consumer of that class.

## Checkpoint — fix scope confirmed — 2026-08-24

User confirmed fixing both the GPU kernel (live bug) and the CPU sibling (identical dormant
bug), for consistency.

## Stage 4 — fix

- `project4d.cu`: threshold now scales with the face's own first-edge length
  (`edge_len2 * edge_len2`, matching n_len2's edge⁴ units), plus a small absolute floor
  (`edge_len2 < 1e-12f`) for genuinely coincident/collinear corners.
- `Mesh4DProjection.scala`: same relative-threshold fix, mirrored (nl scales with edge² here
  since it's the unsquared magnitude, so the edge-length scale is applied without squaring
  twice — see inline comment).
- Reverted the temporary shader-instrumentation debug code from `hit_triangle.cu` before
  committing; cleared the local `~/.ivy2`/coursier publish overrides so menger resolves the
  real fix through a clean `sbt publishLocal`, not stale debug bytes.

**Verification:**
- Visual crosscheck: both repro scenes (straddling horizon, camera-shifted fully above
  horizon) now show correct hole detail continuing across the horizon line, matching the
  user's original screenshots' description.
- optix-jni pre-push hook: full pass (593 Scala/C++ tests, cppcheck, lint, parity, policy).
- Regression-coverage gap found and closed: `integration-tests.sh` had **no successful-render
  test for `tesseract-sponge` above level 1** — level 0/1 faces are too large to ever trip the
  old fallback, so this exact bug class had zero CI coverage. Added a level-2 test.
- Regenerating references surfaced small (~0.001-0.01%) diffs on several *level 1* and
  fractional-level tests too — a few grazing-angle faces near the old absolute threshold were
  also affected, even at shallower recursion. Diffs are confined to silhouette edges and
  glass-refraction boundaries (both normal-dependent), not a structural change. All affected
  references regenerated and verified deterministic across a second identical run (0-pixel
  diff) before committing.
- menger pre-push hook: full run (unit tests, scalafix, packaging, integration suite,
  coverage ratchet, Valgrind/compute-sanitizer).

## Minimum artefacts checklist

- [x] Regression test exists (`integration-tests.sh` "tesseract-sponge level 2", permanent CI
      coverage for the exact recursion depth that triggers the bug class)
- [x] Fix commits pass each repo's full pre-push gate
- [x] This note records the failed first hypothesis (row-based detector confound), the
      user's corrective evidence, and the full root-cause chain
- [x] Prior incorrect "verified resolved, not a bug" note in `ManualTestNeedFixing.md` #8 and
      `SPRINT36.md` corrected
- [x] Scripted integration coverage's reference images regenerated and verified deterministic
- [x] Fixed in both live path (`project4d.cu`) and dormant sibling (`Mesh4DProjection.scala`)
      to prevent the same defect resurfacing if the CPU path is ever exercised
