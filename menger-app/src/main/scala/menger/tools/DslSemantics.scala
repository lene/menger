package menger.tools

import menger.dsl.FieldConstraints

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
    "Units: positions and sizes are world units; `pos` is an object's centre; 3D `rotation` " +
      "angles are radians, but the 4D rotation angles of `Projection4DSpec` (`rotXW`, `rotYW`, " +
      "`rotZW`) are DEGREES (usability review 2026-10, session 3, F74).",
    "Axes: +y is up. The rendered image is currently mirrored horizontally: with the camera " +
      "on +z looking toward -z, +x appears on the LEFT of the image.",
    "Directional light: `direction` is the direction the light TRAVELS. (0f, -1f, 0f) shines " +
      "straight down; (1f, -1f, -1f) comes from above. Point and area lights are placed by " +
      "`position`.",
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
      "(usability review 2026-10, session 3, F66).",
    "Caustics has no intensity parameter (only `photonsPerIteration`, `iterations`, " +
      "`initialRadius`, `alpha`): more photons or iterations reduce noise, they do not make " +
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
      "(600-cell). Each 4D object has its own `projection`, so one object can be rotated in " +
      "4D next to an unrotated one; in the render window, Shift+drag adds the same 4D " +
      "rotation to every 4D object. A 4D object's `size` is applied before the 4D perspective " +
      "projection, so its on-screen size is not proportional to `size` and differs between " +
      "polytope types (a Tesseract and an Icositetrachoron of the same `size` project about " +
      "1.6x apart); to enlarge one without distorting it, scale `size`, `eyeW` and `screenW` " +
      "by the same factor (usability review 2026-10, session 3, F72). The projection scales " +
      "each point by (eyeW - screenW) / (eyeW - w); a flat object (every w = 0, e.g. " +
      "`wScale = 0`) is therefore shown at (eyeW - screenW) / eyeW of its `size` -- 0.5 for " +
      "eyeW 3, screenW 1.5. To match a 3D cube of size s, give the 4D object size " +
      "s * eyeW / (eyeW - screenW) (usability review 2026-10, session 4, F101). " +
      "`Projection4DSpec(wScale = s)` scales the object's w coordinate before the 4D rotation " +
      "(default 1, must be >= 0): 0 flattens a Tesseract to a cube, and animating it from 0 to " +
      "1 grows the tesseract out of the cube along w -- the nearest the DSL has to extruding a " +
      "3D object into 4D. There is no 5D object (menger#65, #59).",
    "Procedural colouring (`proceduralType`) is evaluated at the hit point's world position " +
      "(11 xyz_rgb_local: in the object's own box). " +
      "For a 4D object that is the projected 3D position, after `projection` and `pos`, so the " +
      "colours follow the projected shape and change when the 4D rotation changes " +
      "(usability review 2026-09, session 2, F43); its box is taken from `size` before " +
      "projection, so xyz_rgb_local only approximately spans 0..1 on a 4D object.",
    "Shadows of transparent objects are opaque unless the scene enables them: " +
      "`RenderSettings(transparentShadows = true)`. Then a transparent object casts the colour " +
      "seen through it, its procedural colour included -- e.g. rainbow shadows of a glass " +
      "object with xyz_rgb_local (usability review 2026-10, session 3, F67).",
    "Image textures and texture maps (`texture`, `videoTexture`, `normalMap`, " +
      "`roughnessMap`) need surface UV coordinates. 4D objects, edge tubes (`edgeRadius`), " +
      "curves and an L-system's branches have no UV coordinates, so these fields have no " +
      "effect on them; " +
      "use a material colour or a `proceduralType` there instead.",
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

  /** Per-field notes, keyed by (DSL type's simple name, field name). */
  val fieldDescriptions: Map[(String, String), String] = Map(
    ("Directional", "direction") ->
      "Direction the light travels: (0f, -1f, 0f) shines straight down.",
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
    ("Tesseract", "rotation") -> ("3D Euler angles in radians, applied after projection; " +
      "4D rotations go in `projection`."),
    ("TesseractSponge", "rotation") -> ("3D Euler angles in radians, applied after " +
      "projection; 4D rotations go in `projection`."),
    ("Tesseract", "edgeRadius") -> "Draws the edges as tubes of this radius.",
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
  val holedTypes: Set[String] = Set("Sponge", "TesseractSponge", "Sierpinski4D")

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
