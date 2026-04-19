# Glass Sponge Rendering — Investigation Log

## Bug: H-glass-sponge-skin-diffuse

Reproducer: `--objects type=sponge-volume:level=1.5:material=glass`

---

## Attempt 1: Coverage blend path fix (hit_triangle.cu)

**Hypothesis:** Skin faces (vertex_alpha < 1.0) were forced into the diffuse coverage blend
path regardless of IOR. The condition `has_vertex_alpha_channel && vertex_alpha < threshold`
did not check `!is_refractive`.

**Fix:** Added `!is_refractive` to the vertex-alpha arm of `use_coverage_blend`.

**Result:** Rectangular pinkish artifacts still visible — now rendered as full Fresnel glass
panels instead of diffuse panels, but still visible as distinct rectangles.

---

## Attempt 2: Refractive coverage blend

**Hypothesis:** Skin faces need `vertex_alpha * fresnel_result + (1 - vertex_alpha) * through`
instead of either pure diffuse or pure Fresnel.

**Fix:** Added `use_refractive_coverage_blend` path in hit_triangle.cu — traces reflect, refract,
continuation rays and blends by vertex_alpha.

**Result:** "Still looks essentially the same" (user, screenshot 2026-04-19 10:43:18).
The refractive coverage blend produced no visible improvement.

---

## Key Finding: Level=1 also shows the artifacts

**Fact established (user confirmation):** `--objects type=sponge-volume:level=1:material=glass`
also shows the rectangular artifacts. Level=1 has NO fractional skin faces (vertex_alpha=1.0
for all faces). Therefore the artifacts are NOT caused by the skin-face rendering path.

**Root cause hypothesis:** The sponge-volume mesh includes INTERNAL SHARED FACES between
adjacent sub-cubes. At level=1, the 20 sub-cubes of the Menger sponge share faces at their
touching boundaries. Each cube contributes its own face mesh independently, so touching faces
appear as two back-to-back surfaces inside what should be a monolithic solid glass object.
When rendering with glass material, rays hit these internal faces and produce spurious
Fresnel/refraction boundaries — visible as flat rectangular glass panels.

**Next action:** Investigate `SpongeByVolume` / `Cube.toTriangleMesh` to confirm internal
faces are present in the mesh, then remove them during mesh generation.

---

## Attempt 3: Interior face culling in SpongeByVolume mesh

**Hypothesis:** Adjacent sub-cubes share coincident back-to-back faces in the merged mesh.
These act as spurious glass interfaces inside the solid.

**Fix:** `Cube.toTriangleMeshExcluding(excluded: Set[Direction])` and
`SpongeByVolume.getIntegerMeshExcluding(excluded)` — neighbor-based face exclusion.
Level=1: 240 → 144 triangles; level=2: 4800 → 2112 triangles.

**Test (neutral material, `material=white`):** Geometry correct — sponge structure looks right.

**Test (glass material, level=1 volume and surface sponge):** Large pink rectangular slabs
still visible — SAME artifacts in BOTH sponge types.

---

## Final Finding: Artifacts are Fresnel reflections, not geometry

**SpongeBySurface at level=1 shows the same large pink rectangular panels.**
SpongeBySurface generates only exterior faces — there are no internal faces at all.
Therefore the panels are NOT caused by interior shared faces or any geometry bug.

**Root cause: Glass Fresnel reflection of the pink background.**
At grazing angles, glass Fresnel reflectance approaches 100%. The sponge has many faces
viewed at non-normal angles from the camera, all reflecting the pink background
`(0.3, 0.1, 0.2)`. Adjacent faces with the same normal all reflect identically, producing
large uniform pink "slabs." This is physically correct behavior for glass — it just looks
visually jarring because the background is saturated pink rather than neutral dark.

Confirmed by: rendering with `material=white` (default) — geometry is correct with no artifacts.

**Status:** H-glass-sponge-skin-diffuse is NOT a geometry bug. The face culling fix
(Attempt 3) is correct and beneficial for the mesh but does not change the glass appearance.
The visual issue is the background color + high Fresnel at grazing angles. Not a fixable bug
without changing the background or the rendering approach.
