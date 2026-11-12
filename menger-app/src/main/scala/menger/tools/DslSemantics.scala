package menger.tools

import menger.dsl.FieldConstraints
import menger.dsl.ResourceLimits
import menger.objects.higher_d.NdSponge
import menger.objects.higher_d.NdSpongeVariant

/** What the DSL's fields *mean* -- units, conventions and interactions that reflection cannot
  * recover (there is no scaladoc at runtime). Everything else in the manifest is derived from
  * the DSL types themselves; this is the one deliberately hand-maintained part, so keep it to
  * facts a scene author gets wrong without being told.
  *
  * Added after the scene agent's usability review (2026-09, F28): with names, types and
  * defaults only, the agent lit every scene from below (light direction), made glass opaque
  * (colour alpha) and promised glows the renderer cannot draw.
  */
object DslSemantics:

  /** The highest level of each n-sponge variant per dimension, from the scene's triangle budget
    * as it stands when the manifest is made. */
  private def hypercubeSpongeLevels: String =
    List(
      NdSpongeVariant.Volume -> "VolumeRemoving",
      NdSpongeVariant.Surface -> "SurfaceSubdividing"
    )
      .map { (variant, name) =>
        val perDimension = (NdSponge.MinDimension to NdSponge.MaxDimension).map(n =>
          s"N=$n: ${ResourceLimits.ndSpongeLevel(variant, n).max}"
        ).mkString(", ")
        s"$name at k = 1 $perDimension"
      }.mkString("; ")

  /** Cross-cutting rules that belong to no single field. */
  val conventions: List[String] = List(
    "Scene budget: besides each field's own max, a whole scene must stay within " +
      s"${menger.dsl.ResourceLimits.sceneMaxTriangles} mesh triangles (both levels of a " +
      "fractional sponge count; one VolumeFilling sponge at level 5 fits, two do not), " +
      s"${menger.dsl.ResourceLimits.sceneMaxRendersPerFrame} renders per frame (accumulation " +
      "times 2^k for k fractional-level glass objects, k + 1 above 4) and " +
      s"${menger.dsl.ResourceLimits.sceneMaxPhotonsPerFrame} caustics photons per frame " +
      "(photonsPerIteration x iterations x renders per frame). The validator reports a " +
      "`scene-budget` finding naming what to lower.",
    "Units: positions and sizes are world units; `pos` is an object's centre; an angle-valued " +
      "field names its unit in the manifest's `unit` key. 3D `rotation` angles are radians, but " +
      "the 4D rotation angles of `Projection4DSpec` (`rotXW`, `rotYW`, `rotZW`), " +
      "`LSystem.angleDegrees` and `Camera.fov` are DEGREES (usability review 2026-10, session " +
      "3, F74; menger#95).",
    "Axes: +y is up. The rendered image is currently mirrored horizontally: with the camera " +
      "on +z looking toward -z, +x appears on the LEFT of the image.",
    "Directional light: `direction` is the direction the light TRAVELS. (0f, -1f, 0f) shines " +
      "straight down; (1f, -1f, -1f) comes from above. Point and area lights are placed by " +
      "`position`.",
    "Sun: a distant disk-shaped key light, `Sun(direction, diameterDegrees, distance, color, " +
      "intensity)`. `direction` is the direction the light TRAVELS, as for `Directional`, and " +
      "`intensity` means the same as there (it lights the origin like a `Directional` of that " +
      "intensity, whatever the `distance`); `distance` sets how uniform the light is, not the " +
      "penumbra: shading falls off as 1/(1+r^2) from the disk, exact at the origin, so a point s " +
      "units toward the light is about (1 + 2s/distance) brighter (default 1000: under 1 % " +
      "over +-4 units). Only the FIRST light of a scene emits caustic photons: put the `Sun` first in " +
      "`lights` when the scene has caustics (menger#158).",
    "Planes: `Y at -2f` is an infinite plane that faces the origin, so a floor below the " +
      "scene is lit from above and receives shadows. Shadows are on by default.",
    "Colour and transparency: alpha 0 is fully transparent, 1 fully opaque. An object's " +
      "explicit `color` tints only the material's RGB; the material's own alpha (its " +
      "transparency) is preserved, so an opaque `color` on `Glass` or `Film` stays " +
      "transparent (usability review 2026-09, F14).",
    "Refractive materials (Glass, Diamond, Water, Film): `Material.Glass.copy(color = " +
      "Color(r, g, b))` REPLACES the preset's colour including its alpha, and `Color(r, g, b)` " +
      "has alpha 1, which for a refractive material means fully absorbing -- an opaque look. " +
      "Tint a refractive material with a low alpha, e.g. `Color(0.8f, 0.06f, 0.12f, 0.05f)` " +
      "or, naming what the alpha means there, `Color.tint(0.8f, 0.06f, 0.12f, absorption = " +
      "0.05f)` (for a refractive material the alpha is the ABSORPTION, not the opacity) " +
      "(usability review 2026-10, session 3, F66).",
    "Caustics has no intensity parameter (only `photonsPerIteration`, `iterations`, " +
      "`initialRadius`, `radiusReduction`): more photons or iterations reduce noise, they do not make " +
      "the caustics brighter (usability review 2026-10, session 3, F69). Caustics are only " +
      "visible where a CURVED refracting surface (e.g. a glass Sphere) focuses light onto a " +
      "nearby diffuse surface: flat facets (polytopes, " +
      "sponges) only shift light and thin Film bends it hardly at all, so they give faint or " +
      "no caustics -- say so instead of promising patterns (usability review 2026-10, " +
      "session 4, F98). Every transparent material with an IOR above 1 refracts, Film " +
      "(1.33) and Water included, not only Glass and Diamond (session 5, F121).",
    "Emission makes a surface self-lit: flat, unshaded colour. There is no bloom, halo or " +
      "glow around objects, and no emission that falls off with distance.",
    "Animation: `def scene(t: Float): Scene` plus `val durationSeconds = <seconds>f` in the " +
      "same object; t is seconds in [0, durationSeconds]. The window plays it in real time, " +
      s"looping, and the validator checks the scene at ${SceneValidator.AnimationSampleCount} " +
      "evenly spaced times from t = 0 to " +
      "t = durationSeconds. The " +
      "older name `val duration` is deprecated but still read. An animated scene has no " +
      "`SceneRegistry.register` line: register takes a static Scene only, so drop it when " +
      "turning `val scene` into `def scene(t: Float)` (usability review 2026-09, F46). " +
      "The scene has no frame count. The scene agent's window plays it with every frame " +
      "rendered (25 steps per second at most; a slow frame slows the playback instead of " +
      "skipping scene time) and loops; menger's default real-time mode follows the wall " +
      "clock and skips scene time when rendering is slow, which the window title says. A " +
      "playing window with no input for 10 minutes pauses itself (Space resumes); frame " +
      "counts are menger-app command-line options only. Editing the scene file reloads an animated scene live in " +
      "the window, but the window cannot switch between a static and an animated scene -- " +
      "it must be restarted then (usability review 2026-10, session 3, F57, F84). The window " +
      "has playback keys: Space pauses and resumes, Left/Right step 0.01 s (with Shift " +
      "0.1 s), Home/End jump to the start/end -- so a looping animation CAN be stopped at any " +
      "frame (usability review 2026-10, session 4, F92).",
    "4D objects are rotated in 4D (`projection`), projected to 3D, then placed at `pos`. " +
      "The regular 4D polytopes are Pentachoron (5-cell), Tesseract (8-cell), Hexadecachoron " +
      "(16-cell), Icositetrachoron (24-cell), Hecatonicosachoron (120-cell) and Hexacosichoron " +
      "(600-cell). The 4D IFS fractals are Menger4D (the TesseractSponge's shape), " +
      "PentachoronGasket and HexadecachoronGasket (Sierpinski gaskets of the 5-cell and the " +
      "16-cell); they are traced on the GPU without a stored mesh, so a higher level costs " +
      "render time, not memory. Each 4D object has its own `projection`, so one object can be rotated in " +
      "4D next to an unrotated one; in the render window, Shift+drag adds the same 4D " +
      "rotation to every 4D object. A 4D object's `size` is applied before the 4D perspective " +
      "projection, so its on-screen size is not proportional to `size` and differs between " +
      "polytope types (a Tesseract and an Icositetrachoron of the same `size` project about " +
      "1.6x apart); to enlarge one without distorting it, scale `size`, `eyeW` and `screenW` " +
      "by the same factor (usability review 2026-10, session 3, F72). The projection scales " +
      "each point by (eyeW - screenW) / (eyeW - w); a flat object (every w = 0, e.g. " +
      "`scale = Some(Scale[4](1f, 1f, 1f, 0f))`) is therefore shown at (eyeW - screenW) / " +
      "eyeW of its `size` -- 0.5 for eyeW 3, screenW 1.5. To match a 3D cube of size s, give " +
      "the 4D object size s * eyeW / (eyeW - screenW) (usability review 2026-10, session 4, " +
      "F101). A `ProjectionSpec` (the nD view convention below) lifts a 4D object into 5D to 8D " +
      "(menger#59, #162).",
    "Per-axis scale: the 4D objects (Tesseract, the regular polytopes, TesseractSponge) take " +
      "`scale = Some(Scale[4](x, y, z, w))` and Hypercube takes `Scale[N]` for its own " +
      "dimension N: one factor >= 0 per axis, finite, applied to the object's coordinates " +
      "after `size` and before the rotations and the projection (default all 1; a wrong " +
      "number of factors or a negative one is rejected when the scene loads). 0 collapses an " +
      "axis, so the last factor of an object is its extrusion along the newest axis: " +
      "animating it 0 -> 1 grows a Tesseract out of a cube, or a Hypercube(5) out of a " +
      "tesseract. `scale` is geometry, not view: a change rebuilds the object, where a change " +
      "of `projection` re-projects it. The GPU IFS types (Menger4D, PentachoronGasket, " +
      "HexadecachoronGasket) and the 3D objects have no `scale`. On the command line: " +
      "`scale=1,1,1,0.5` (menger#65, #164).",
    "Hand-over rule: a Hypercube(n) whose last `scale` factor is 0 is the Hypercube(n - 1), " +
      "seen in the same n-dimensional view: the same vertices, and its faces that use the " +
      "collapsed axis shrink to lines. So an animation can show the lower object, then the " +
      "higher one growing out of it, with no cut. Mind four points. (1) In its own " +
      "(n - 1)-view the lower object is bigger than the flat higher one, which is shown at " +
      "(eye - screen) / eye of its size (0.5 by default), so swap exactly at scale 0 and " +
      "give the higher object size s * eye / (eye - screen) of its first projection stage for " +
      "a lower object of size s; the 3D Cube -> 4D case is the same rule (Hypercube(4) or " +
      "Tesseract with a w scale, size s * eyeW / (eyeW - screenW)). (2) Every rotation plane " +
      "that involves the collapsing axis must be at a multiple of 180 degrees when it " +
      "collapses, or the flat object is a turned slice, not the lower object: keep those " +
      "planes at 0 until the scale is above 0, or turn a whole 360 degrees. (3) Near 0 the " +
      "two copies of each face nearly coincide: expect z-fighting for a few frames, and " +
      "show the lower object at exactly 0 (a transparent material shows the coincident " +
      "copies as speckle). (4) Give both objects the same material and the same " +
      "`edgeRadius`/`edgeMaterial` (a `material`, not only a `color`: a 4D object with edge " +
      "tubes and no material is drawn as its edges alone), and let the extrusion take at " +
      "least 3 s; edge tubes of zero length are not drawn (usability review 2026-10, " +
      "session 5, F124, menger#132, #164; examples: CubeToTesseract, PenteractFold).",
    "nD view: `projection = Some(ProjectionSpec(dimension, planes, eye, screen, stages))` " +
      "lifts a 4D object (Tesseract, TesseractSponge or a regular polytope) into `dimension` " +
      "4..8 dimensions: its x y z w coordinates stay, the further ones are 0. The axes are " +
      "x y z w v u t s (5D adds v, 6D u, 7D t, 8D s). `planes` is a list of " +
      "`PlaneRotation(plane, angle)` with the plane named by two distinct axes that exist in " +
      "`dimension`, e.g. `PlaneRotation(\"xv\", 30f)` (the angle is in DEGREES; for a " +
      "positive angle the second axis turns towards the first, v towards x in `xv`, as w " +
      "does in `Projection4DSpec.rotXW`), applied in the order written: " +
      "the first plane turns first, and the result depends on the order. An object lies in " +
      "x y z w, so a plane of axes beyond w alone (vu, ...) turns nothing until an earlier " +
      "plane (xv, wv, ...) has moved the object into them. The projection to 3D has one " +
      "perspective stage per dropped axis, the first dropping the highest axis; a stage " +
      "scales a point by (eye - screen) / (eye - c), c the coordinate it drops, so the stages " +
      "compound: with the default eye 3 and screen 1.5 a flat object is at 0.5^(dimension - " +
      "3) of its `size`, half again per extra dimension -- enlarge `size`, or lower `screen`, " +
      "to compensate. `stages` may list one `ProjectionStage(eye, " +
      "screen)` per stage (dimension - 3 of them); empty, every stage uses `eye` and " +
      "`screen`. Keep each eye above the extent of the lifted object along the axis it " +
      "drops: faces of a 4D object reaching an eye are not clipped and distort (a Hypercube's " +
      "are left out), edge tubes with such an " +
      "endpoint are left out. A `ProjectionSpec` is rendered on the CPU through the chain: " +
      "an animation rebuilds the object every frame, a sponge needs a whole `level`, the " +
      "IFS types (Menger4D, PentachoronGasket, HexadecachoronGasket) take a " +
      "`Projection4DSpec` only, and the render window's Shift+drag and Shift+scroll (eye) " +
      "leave it alone (the log says so once). The object's `scale` applies before the lift, " +
      "so a scaled 4D object can be lifted into 5D or more. " +
      "`ProjectionSpec(4, ...)` is the 4D view with the planes in " +
      "order: `Projection4DSpec(rotXW = a, rotYW = b, rotZW = c)` equals planes zw c, yw b, " +
      "xw a, in that order (menger#162; example: TesseractToPenteract).",
    "Hypercube: `Hypercube(dimension = N)` is the cube of N = 4..8 dimensions (4 the " +
      "tesseract, 5 the penteract, 6 the hexeract, ...) with 2^N vertices, N * 2^(N-1) edges " +
      "and C(N,2) * 2^(N-2) square faces; `size` is its edge length, its centre is `pos`. It " +
      "takes a `ProjectionSpec` of `dimension` N or more, as `projection = Some(...)`; absent, " +
      "`ProjectionSpec(N)` with its defaults. A cube of fewer dimensions than the view is " +
      "lifted into it with zero coordinates, so planes using the extra axes act on it. It " +
      "is rendered on the CPU and rebuilt for every frame of an animation; a face or edge tube " +
      "with a vertex at or behind a stage's eye is left out, the rest is drawn. With " +
      "`edgeRadius`/`edgeMaterial` it has one tube per edge: 1024 on an 8-cube, which the " +
      "renderer's instance limit is raised for. Hypercube is DSL-only: the command line has " +
      "only the 4D `tesseract` (menger#163; example: HexeractTurning).",
    "HypercubeSponge: `HypercubeSponge(dimension = N, spongeType, level, " +
      "maxCentreCoordinates = k)` is the Menger sponge of N = 4..8 dimensions, built from the " +
      "N-cube by splitting every axis in thirds, `level` times over. `VolumeRemoving` keeps " +
      "the sub-cubes with at most k coordinates in the middle third: k = 1 (default) is rule " +
      "A, the generalisation of the TesseractSponge, k = N - 2 is rule B, which removes only " +
      "the middle cell and its axis neighbours, and k may be 1..N - 1. `SurfaceSubdividing` " +
      "is rule S: every square face splits into nine and its middle one is replaced by walls " +
      "one square deep around a hole, along each of the N - 2 normals of the face; with a " +
      "`Film` material it looks calmer than the volume sponge. Level 0 is the n-cube " +
      "with the same view and size, so an animation can hand over to `Hypercube(N)` there " +
      "(the sponge itself has no edge tubes). The " +
      "highest `level` follows the scene's triangle budget and so the dimension: " +
      hypercubeSpongeLevels + ". A whole level costs 2 triangles per face " +
      "(surface, N = 5, level 2: 28,160 faces); a level that fits the " +
      "budget is accepted, one over it is refused with the level that fits. It takes a " +
      "`ProjectionSpec` of dimension N or more as `projection = Some(...)` (absent: " +
      "`ProjectionSpec(N)`) and a `Scale[N]` as `scale`. It is rendered on the CPU: " +
      "the unprojected sponge is built once and kept, so an animation that only turns, " +
      "scales or re-views it projects it again each frame, but one that changes the level " +
      "builds a sponge per whole level. A fractional level fades the new holes in over the " +
      "caps of the lower level (opaque materials) or mixes the two whole levels (refractive " +
      "ones), as the 4D sponges do, and is allowed with a `ProjectionSpec`. DSL-only (the " +
      "command line has the 4D sponges); no edge tubes (menger#166; example: " +
      "PenteractSpongeBreath).",
    "Procedural colouring (`proceduralType`) is evaluated at the hit point's world position " +
      "(11 xyz_rgb_local: in the object's own box). " +
      "For a 4D object that is the projected 3D position, after `projection` and `pos`, so the " +
      "colours follow the projected shape and change when the 4D rotation changes " +
      "(usability review 2026-09, session 2, F43); its box is taken from `size` before " +
      "projection, so xyz_rgb_local only approximately spans 0..1 on a 4D object.",
    "Shadows of transparent objects: `RenderSettings(transparentShadows = ...)` is `Auto` by " +
      "default, which makes a transparent object cast the colour seen through it when one is " +
      "tinted or procedurally coloured -- e.g. rainbow shadows of a glass object with " +
      "xyz_rgb_local (usability review 2026-10, session 3, F67; session 5, F111). `true` " +
      "turns it on for every transparent object, `false` keeps shadows opaque. A scene without " +
      "`render = Some(RenderSettings(...))` behaves as `Auto`. `caustics` turns transparent " +
      "shadows on in any case.",
    "Image textures and texture maps (`texture`, `videoTexture`, `normalMap`, " +
      "`roughnessMap`) need surface UV coordinates. 4D objects, edge tubes (`edgeRadius`), " +
      "a `Cylinder`, curves and an L-system's branches have no UV coordinates, so these " +
      "fields have no effect on them; " +
      "use a material colour or a `proceduralType` there instead.",
    "A `Cylinder` is shaded opaque, diffuse or metallic: a transparent material (Glass, " +
      "Water, Diamond, Film, Parchment, or any colour alpha below 1) renders opaque, a material " +
      s"with an IOR above ${menger.engines.scene.LevelMixing.RefractiveIor} (the refractive " +
      "ones, and Plastic) receives no caustic, and thin film (`filmThickness`) has no effect " +
      "on it. The validator reports either as an " +
      s"`${SceneValidator.OpaqueOnlyInvariant}` finding; Matte and the metals are fine. " +
      "Object-local procedural colouring (`proceduralType` 11, xyz_rgb_local) spans the box " +
      "`pos` +- `size / 2`, not `p0`/`p1`: with `p0`/`p1`, set `pos` and `size` so that box " +
      "covers the cylinder. A wide, flat Cylinder makes a round plinth or table top " +
      "(menger#156).",
    "Glass on a TesseractSponge from level 2 up renders as chaotic, fragmented refraction: the " +
      "projected sponge has thousands of overlapping refracting layers and a ray gets at most 5 " +
      "bounces. Prefer an opaque or metal material there, or level 1 for glass, and say so if " +
      "the user asks for glass (usability review 2026-09, session 2, F55). Thin transparent " +
      "`Film` shows every inner layer of a 4D sponge at once, so its holes and their walls " +
      "can't be told apart from the front face; an opaque material with a directional light " +
      "shows the hole shapes (menger#56).",
    "Camera: the horizontal field of view is fixed at 45 degrees (not adjustable per scene). " +
      "To frame or zoom to fit a scene, move `Camera.position` -- aim `lookAt` at the scene's " +
      "centre and set the eye's distance from it to at least radius / sin(22.5deg), where " +
      "radius is the scene's bounding-sphere radius, with a margin for comfortable framing."
  )

  private val Scale4DDescription =
    "Four factors `Scale[4](x, y, z, w)`, each >= 0, applied after `size` and before the 4D " +
      "rotation; 0 collapses an axis, the last factor is the extrusion along w: 0 flattens " +
      "the object to 3D, animating it 0 -> 1 grows it out of that. See the hand-over rule in " +
      "conventions."

  /** Per-field notes, keyed by (DSL type's simple name, field name). */
  val fieldDescriptions: Map[(String, String), String] = Map(
    ("Directional", "direction") ->
      "Direction the light travels: (0f, -1f, 0f) shines straight down.",
    ("Sun", "direction") ->
      "Direction the light travels (as Directional): (0f, -1f, 0f) shines straight down.",
    ("Sun", "diameterDegrees") ->
      "Apparent diameter of the sun's disk in degrees, above 0 and at most 10; the sun is 0.53.",
    ("Sun", "distance") ->
      "Distance of the disk from the origin (default 1000); larger = more uniform light across the scene, the penumbra stays.",
    ("Sun", "intensity") -> "Brightness, as for Directional.",
    ("Plane", "axisPosition") ->
      "`Y at -2f` = floor at y = -2 facing up (planes face the origin).",
    ("Camera", "position") -> "Eye position in world units.",
    ("Camera", "lookAt") -> ("Point the camera looks at; frame a scene by aiming here at the " +
      "centre of all objects and moving `position` away until they fit."),
    ("Sponge", "level") -> ("Recursion depth; fractional levels blend between the two " +
      "neighbouring integer levels. In a refractive material (glass, water, diamond, film) the " +
      "renderer renders both levels and mixes the images, so each such object doubles the " +
      "renders per frame (two objects 4, up to 16) (menger#69). Cost grows ~20x per level. Limits depend on `spongeType` " +
      "(see `limitsBy`): VolumeFilling and SurfaceUnfolding go up to 5 (a VolumeFilling level 5 " +
      "is a 13M-triangle mesh, about 10 s to build); RecursiveIAS covers levels 1 to 13."),
    ("TesseractSponge", "level") -> ("Recursion depth; fractional levels blend between " +
      "integer levels. Cost grows ~48x per level."),
    ("TesseractSponge", "size") -> "Scale; 1 matches a Tesseract of size 1.",
    ("Sphere", "size") -> "Radius in world units.",
    ("Sponge", "color") -> "Tints the material's RGB; alpha (transparency) is preserved (see conventions).",
    ("TesseractSponge", "color") -> "Tints the material's RGB; alpha (transparency) is preserved.",
    ("Tesseract", "color") -> "Tints the material's RGB; alpha (transparency) is preserved.",
    ("Sphere", "color") -> "Tints the material's RGB; alpha (transparency) is preserved.",
    ("Sponge", "rotation") -> "Euler angles in radians around x, y, z.",
    ("Tesseract", "scale") -> Scale4DDescription,
    ("TesseractSponge", "scale") -> Scale4DDescription,
    ("Pentachoron", "scale") -> Scale4DDescription,
    ("Hexadecachoron", "scale") -> Scale4DDescription,
    ("Icositetrachoron", "scale") -> Scale4DDescription,
    ("Hexacosichoron", "scale") -> Scale4DDescription,
    ("Hecatonicosachoron", "scale") -> Scale4DDescription,
    ("Hypercube", "scale") -> ("One factor >= 0 per axis, `Scale[N]` for the cube's dimension N " +
      "(a wrong count is rejected), applied after `size`; 0 collapses an axis, the last factor " +
      "is the extrusion along the newest axis: Hypercube(n) with the last factor 0 is the " +
      "Hypercube(n - 1). See the hand-over rule in conventions."),
    ("Hypercube", "dimension") -> ("Number of dimensions N, 4 to 8: the cube has 2^N " +
      "vertices; see conventions."),
    ("HypercubeSponge", "dimension") -> ("Number of dimensions N, 4 to 8; see conventions " +
      "for the highest level of each."),
    ("HypercubeSponge", "spongeType") -> ("VolumeRemoving keeps the sub-cubes with at most " +
      "`maxCentreCoordinates` middle coordinates (rule A for 1); SurfaceSubdividing is rule " +
      "S, walls one square deep around the holes."),
    ("HypercubeSponge", "level") -> ("0 is the n-cube; the maximum depends on the dimension " +
      "and the variant (see conventions); a fractional level fades the new holes in."),
    ("HypercubeSponge", "maxCentreCoordinates") -> ("k, 1..N-1: the most middle-third " +
      "coordinates a kept sub-cube may have; 1 = rule A (default), N-2 = rule B. Higher k " +
      "keeps more and costs far more faces, so a level that fits at k = 1 may not at k = 3."),
    ("HypercubeSponge", "scale") -> ("One factor >= 0 per axis, `Scale[N]` for the sponge's " +
      "dimension N (a wrong count is rejected), applied after `size`."),
    ("HypercubeSponge", "color") -> "Tints the material's RGB; alpha (transparency) is preserved.",
    ("HypercubeSponge", "rotation") -> ("3D Euler angles in radians, applied after " +
      "projection; rotations in N dimensions go in `projection`."),
    ("Hypercube", "color") -> "Tints the material's RGB; alpha (transparency) is preserved.",
    ("Hypercube", "rotation") -> ("3D Euler angles in radians, applied after projection; " +
      "rotations in N dimensions go in `projection`."),
    ("Hypercube", "edgeRadius") -> "Draws the edges as tubes of this radius, N * 2^(N-1) of them.",
    ("Tesseract", "rotation") -> ("3D Euler angles in radians, applied after projection; " +
      "4D rotations go in `projection`."),
    ("TesseractSponge", "rotation") -> ("3D Euler angles in radians, applied after " +
      "projection; 4D rotations go in `projection`."),
    ("Cylinder", "p0") -> ("Centre of one end cap, in world coordinates. Give it together " +
      "with `p1` (both or neither); they override `pos`, `size` and `rotation`."),
    ("Cylinder", "p1") -> ("Centre of the other end cap, in world coordinates; together with " +
      "`p0` (both or neither)."),
    ("Cylinder", "radius") -> "Radius in world units; `size / 2` when absent.",
    ("Cylinder", "size") -> ("Height along y (turned by `rotation`, centred on `pos`) when " +
      "`p0`/`p1` are absent; the default radius is `size / 2`."),
    ("Cylinder", "rotation") -> ("Euler angles in radians; turn the default axis (y) about " +
      "`pos`. Ignored when `p0`/`p1` are given."),
    ("Cylinder", "material") -> ("Only opaque, non-refractive materials (alpha 1, IOR " +
      s"${menger.engines.scene.LevelMixing.RefractiveIor} or less, e.g. Matte or a metal): " +
      "see conventions."),
    ("Tesseract", "edgeRadius") -> "Draws the edges as tubes of this radius.",
    ("Cube", "edgeRadius") -> ("Draws the 12 edges as tubes of this radius (`edgeMaterial`, " +
      "default Film) -- the same tubes as on the 4D types (menger#132)."),
    ("Sponge", "edgeRadius") -> ("Draws the edges as tubes of this radius (`edgeMaterial`, " +
      "default Film): where faces meet at an angle, so the grid between coplanar sub-faces is " +
      "not drawn. VolumeFilling and SurfaceUnfolding only (RecursiveIAS has no flat mesh). One " +
      "tube per edge: 96 at level 1, 1296 at level 2, 22512 at level 3; level 4 needs more " +
      "instances than the renderer has. A fractional level blends the tubes of its two levels " +
      "up to 2.x, like the 4D types (menger#132)."),
    ("TesseractSponge", "edgeRadius") -> ("Draws the edges as tubes of this radius. A " +
      "fractional level blends the tubes of its two levels up to 2.x; edge tubes fit only up to " +
      "level 2: one tube per edge, so level 3 needs 130177 instances, more than the 65536 the " +
      "renderer has, and from 2.x the level stays at its integer part (usability review 2026-10, " +
      "session 4, F94; menger#70).")
  )

  /** The same on every object that has them (usability review 2026-09, session 2, F37/F40):
    * by field name, so a new object type gets them without a list to extend (it was a hand list
    * of 14 types). The presets themselves are `ProceduralPresets` (menger#81). */
  private val descriptionsByFieldName: Map[String, String] = Map(
    "proceduralType" -> menger.dsl.ProceduralPresets.description,
    "proceduralScale" -> ("Multiplies the world position before the pattern is evaluated " +
      "(higher = smaller, more frequent pattern). One pattern period across an object needs " +
      "about 1 / size.")
  )

  def descriptionOf(typeName: String, fieldName: String): Option[String] =
    fieldDescriptions.get((typeName, fieldName)).orElse(descriptionsByFieldName.get(fieldName))

  /** Enum values the manifest leaves out, so the agent never chooses them; the DSL still reads
    * them (`CubeSponge`: deprecated, menger#76). */
  val deprecatedEnumValues: Set[String] = Set("CubeSponge")

  /** Object types with holes through them: something inside one stays visible through the holes,
    * so it never occludes (usability review 2026-10, F78). The agent kept this list itself
    * (review 2026-10-14, msa#28); the manifest now carries it as the `holed` trait. */
  val holedTypes: Set[String] =
    Set("Sponge", "TesseractSponge", "HypercubeSponge", "PentachoronGasket", "Menger4D",
      "HexadecachoronGasket")

  // Bounds are `menger.dsl.FieldConstraints` (review 2026-10-14, menger#114): these were three
  // string-keyed tables parallel to the DSL `require`s. `Sponge`/`TesseractSponge` have a
  // ceiling per `spongeType`; their plain max is the smaller of the mesh types' -- never wrong
  // to reject, only sometimes more conservative than the per-variant limit.
  def limitsOf(typeName: String, fieldName: String): (Option[Double], Option[Double]) =
    FieldConstraints.fields.get((typeName, fieldName)).fold((None, None))(b => (b.min, b.max))

  /** Level at or above which rendering gets slow -- the agent should ask before going there
    * (usability review 2026-09, session 2, F36). */
  def warnAtOf(typeName: String, fieldName: String): Option[Double] =
    FieldConstraints.fields.get((typeName, fieldName)).flatMap(_.warnAt)

  def limitsBySubtypeOf(typeName: String, fieldName: String): Option[FieldConstraints.BySubtype] =
    FieldConstraints.bySubtype.get((typeName, fieldName))
