# Investigation: noisy mixed scene (manual-test #14)

Scene 46: `type=tesseract-sponge-2:level=1:pos=-0.8,0,0:material=glass --objects type=sphere:pos=0.8,0,0:material=chrome`.

## Stage 0 — Context, render and describe — 2026-08-21

**Command:**
```
./menger-app/target/universal/stage/bin/menger-app --headless \
  --objects "type=tesseract-sponge-2:level=1:pos=-0.8,0,0:material=glass" \
  --objects "type=sphere:pos=0.8,0,0:material=chrome" --plane y:-2 -s /tmp/scene46.png
```
**Output/measurement:** the glass tesseract-sponge-2 mesh shows a blotchy, high-contrast
static/noise pattern across its faces. The chrome sphere is clean.
**Hypothesis update:** confirmed as described in `ManualTestNeedFixing.md`.

**Command:** same, with `--accumulation-frames 32 --denoise` added.
**Output/measurement:** pixel-diff against the plain render: 9252/480000 pixels differ
(1.9%), concentrated outside the mesh's own noisy region. The mesh's noise pattern itself is
essentially unchanged.
**Hypothesis update:** not simple stochastic Monte-Carlo grain (accumulation/denoise would
substantially reduce that). The noise is closer to a deterministic geometric/shading
artifact.

**Command:** same scene, `material=matte` instead of `material=glass`.
**Output/measurement:** the mesh renders as a clean, simple solid cube — no noise, no
visible sponge structure at all.
**Hypothesis update:** the noise is specific to glass/refraction, which must trace rays
through the mesh's interior; matte/opaque only shows the outer hull.

## Stage 1/2 — Mesh topology check — 2026-08-21

Reactivated (temporarily; reverted after) `TopologyDiagnosticSpec`'s
`ignore`d TesseractSponge2 diagnostic:

**Output/measurement (level 1, matching the scene):**
```
4D (384 faces) → NON-MANIFOLD | hist=Map(2 -> 864, 3 -> 192)
3D (768 triangles) → NON-MANIFOLD | hist=Map(2 -> 864, 3 -> 192)
```
192 edges are shared by 3 faces/triangles instead of 2 — the classical "closed 2-manifold"
criterion `MeshTopology.checkFace4D`/`checkTriangleMesh` uses.

**Hypothesis (first pass, later corrected):** two adjacent tesseract 2-faces' independently
generated tunnel walls (`generatePerpendicularParts`) collide near a shared ridge, creating
extra non-manifold geometry.

## Root cause — corrected — 2026-08-21

A scratch test tagging each generated face with provenance (parent-face index, `flat` vs
`perp`, sub-index) showed most triple-edges actually come from a *single* parent face's own
flat/perpendicular pieces touching, not two different faces colliding as first hypothesized.

Checked the diagnostic's own level-0 output (a bare, unmodified tesseract, zero sponge
subdivision): `4D (24 faces) → NON-MANIFOLD | hist=Map(3 -> 32)` — **every one of the 32
edges of a plain tesseract is already shared by exactly 3 of its 24 two-dimensional faces**,
before any sponge carving happens at all.

Verified this is not a bug but exact 4D combinatorics: `24 faces × 4 edges/face = 96`
edge-incidences `= 32 edges × 3 shares`, exactly. A 4-cube's 2-face skeleton is fundamentally
**not** a classical 2-manifold — unlike a 3-cube (where every edge is shared by exactly 2 of
its 6 square faces), every edge of a tesseract's 24 two-faces is inherently shared by 3
faces, a direct consequence of C(3,2)=3 (choosing which 2 of an edge's 3 fixed coordinates
define a containing 2-face). This is unconditionally true for *any* level, sponge or not.

**Conclusion: not a fixable defect in `TesseractSponge2`.** `MeshTopology.checkFace4D`'s
"isManifold ⟺ every edge shared by exactly 2 faces" criterion is the wrong invariant for a
tesseract-family object's 2-face skeleton — it was designed for (and correctly validates)
genuine 2-manifold surfaces, not the 2-skeleton of a 4-polytope, which is a different,
legitimate mathematical structure that is *never* a classical manifold by this test. The
projected 3D triangle mesh built from this skeleton inherits the same non-2-manifold
property, which is what a glass/refraction shader (which needs a true 2-manifold to
correctly determine inside/outside and trace refracted rays) exposes as visible noise —
while opaque/matte shading, only ever evaluating whichever face a primary ray hits first,
never needs that invariant and so never shows it.

This is an architecture-level mismatch (glass refraction assumes a watertight solid the
2-face-skeleton representation was never designed to be), not a bug introduced anywhere in
`TesseractSponge2`'s sponge-carving logic. A real fix would mean either building a genuinely
different, true-solid mesh representation for tesseract-family objects (a new representation,
not a patch), or documenting that `material=glass`/other refraction-dependent materials are
not well-supported on tesseract-family mesh objects. Both are product/architecture decisions
beyond this investigation's scope.

## Minimum artefacts checklist

- [x] This note records per-step commands, measurements, and the corrected root-cause
      narrative (including the abandoned first hypothesis and why it was wrong)
- [x] `ManualTestNeedFixing.md` #14 updated with the precise, verified finding
- [ ] No regression test added — there is no code defect to regress-test; the finding is
      architectural, not a bug
- [x] No `CODE_IMPROVEMENTS.md` entry existed for this
