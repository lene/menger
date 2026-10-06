package menger.tools

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
      "the caustics brighter (usability review 2026-10, session 3, F69).",
    "Emission makes a surface self-lit: flat, unshaded colour. There is no bloom, halo or " +
      "glow around objects, and no emission that falls off with distance.",
    "Animation: `def scene(t: Float): Scene` plus `val durationSeconds = <seconds>f` in the " +
      "same object; t is seconds in [0, durationSeconds]. The window plays it in real time, " +
      "looping, and the validator checks the scene at t = 0 and at t = durationSeconds. The " +
      "older name `val duration` is deprecated but still read. An animated scene has no " +
      "`SceneRegistry.register` line: register takes a static Scene only, so drop it when " +
      "turning `val scene` into `def scene(t: Float)` (usability review 2026-09, F46). " +
      "The scene has no frame count: the window plays by the wall " +
      "clock and drops frames when rendering is slow; frame counts are menger-app " +
      "command-line options only. Editing the scene file reloads an animated scene live in " +
      "the window, but the window cannot switch between a static and an animated scene -- " +
      "it must be restarted then (usability review 2026-10, session 3, F57, F84).",
    "4D objects are rotated in 4D (`projection`), projected to 3D, then placed at `pos`. " +
      "The regular 4D polytopes are Pentachoron (5-cell), Tesseract (8-cell), Hexadecachoron " +
      "(16-cell), Icositetrachoron (24-cell), Hecatonicosachoron (120-cell) and Hexacosichoron " +
      "(600-cell). Each 4D object has its own `projection`, so one object can be rotated in " +
      "4D next to an unrotated one; in the render window, Shift+drag adds the same 4D " +
      "rotation to every 4D object. A 4D object's `size` is applied before the 4D perspective " +
      "projection, so its on-screen size is not proportional to `size` and differs between " +
      "polytope types (a Tesseract and an Icositetrachoron of the same `size` project about " +
      "1.6x apart); to enlarge one without distorting it, scale `size`, `eyeW` and `screenW` " +
      "by the same factor (usability review 2026-10, session 3, F72). " +
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
      "neighbouring integer levels. Cost grows ~20x per level. Limits depend on `spongeType` " +
      "(see `limitsBy`): VolumeFilling above level 4 builds a mesh too large for the window's " +
      "memory (4.75 ran it out of memory) -- prefer SurfaceUnfolding, or RecursiveIAS (levels " +
      "1 to 13) for high levels."),
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
    ("TesseractSponge", "edgeRadius") -> "Draws the edges as tubes of this radius."
  ) ++ proceduralDescriptions

  /** Every DSL object type has `proceduralType`/`proceduralScale`; the presets were known only
    * to the scene agent's prompt (usability review 2026-09, session 2, F37/F40). Numbers and
    * names match `ObjectSpec`'s preset table and optix-jni's `applyProceduralTexture`. */
  private def proceduralDescriptions: Map[(String, String), String] =
    val objectTypes = List(
      "Sphere", "Cube", "Sponge", "Tesseract", "TesseractSponge", "Sierpinski4D",
      "ParametricSurface", "Curve", "LSystem", "Pentachoron", "Hexadecachoron",
      "Icositetrachoron", "Hexacosichoron", "Hecatonicosachoron"
    )
    val typeText =
      "Built-in procedural texture, one of exactly these presets: 0 none (default); " +
        "1 value_noise, 2 fbm, 3 worley, 4 gradient, 5 wood, 6 marble, 7 layered_noise, " +
        "10 triplanar -- each MODULATES the material's own colour by a pattern (same hue, " +
        "varying brightness); 8 xyz_rgb REPLACES the colour with (|x|, |y|, |z| mod 1) of the " +
        "world position as RGB, mirrored at 0 on each axis; 11 xyz_rgb_local REPLACES it with " +
        "the position inside the object's own box, x, y, z each 0..1 -> R, G, B: every colour " +
        "once over the object at scale 1, following its position, size and rotation, no " +
        "mirroring -- use it for \"local\" or \"every colour once\" colouring, never move the " +
        "object for it (usability review 2026-10, session 3, F63); 9 heatmap REPLACES the " +
        "colour with a blue-to-red noise gradient. It belongs to the look of the material it " +
        "imitates: swapping the material (e.g. wood to aluminium) drops a pattern like wood."
    val scaleText =
      "Multiplies the world position before the pattern is evaluated (higher = smaller, " +
        "more frequent pattern). One pattern period across an object needs about 1 / size."
    objectTypes.flatMap(t =>
      List((t, "proceduralType") -> typeText, (t, "proceduralScale") -> scaleText)
    ).toMap

  def descriptionOf(typeName: String, fieldName: String): Option[String] =
    fieldDescriptions.get((typeName, fieldName))

  /** Per-field (min, max), keyed the same way as [[fieldDescriptions]] -- the single source is
    * [[menger.dsl.ResourceLimits]] (usability review 2026-09, T1#3); this just maps a DSL
    * type/field pair onto the right constant. `Sponge` has one level ceiling regardless of
    * `spongeType` (cube and cube-sponge share `cubeSpongeMaxLevel`); `TesseractSponge` does
    * not (volume-removing and surface-subdividing have different ceilings), so its manifest
    * max is the *smaller* of the two -- a hint that's never wrong to reject, only sometimes
    * more conservative than the real per-variant limit `require()` actually enforces.
    */
  private val fieldLimits: Map[(String, String), (Option[Double], Option[Double])] = Map(
    ("Sponge", "level") -> (Some(0d), Some(menger.dsl.ResourceLimits.cubeSpongeLevel.max.toDouble)),
    ("TesseractSponge", "level") -> (Some(0d), Some(math.min(
      menger.dsl.ResourceLimits.tesseractSpongeVolumeLevel.max,
      menger.dsl.ResourceLimits.tesseractSpongeSurfaceLevel.max
    ).toDouble)),
    ("LSystem", "iterations") -> (Some(0d), Some(menger.dsl.ResourceLimits.lsystemMaxIterations.toDouble)),
    ("ParametricSurface", "uSteps") -> (Some(1d), None),
    ("ParametricSurface", "vSteps") -> (Some(1d), None),
    ("Sierpinski4D", "level") -> (Some(0d), Some(menger.dsl.ResourceLimits.ifs4dMaxLevel.toDouble))
  )

  /** Per-value bounds of a field whose limits depend on another field (usability review
    * 2026-10, session 3, F59): (field the limits are keyed by) -> value -> (min, max, warnAt).
    * The plain `limitsOf`/`warnAtOf` stay as the conservative default for when that field is
    * not a literal. */
  type Bounds = (Option[Double], Option[Double], Option[Double])

  private def meshSpongeBounds: Bounds =
    val limit = menger.dsl.ResourceLimits.cubeSpongeLevel
    (Some(0d), Some(limit.max.toDouble), Some(limit.warnAt.toDouble))

  private def tesseractSpongeBounds(limit: menger.dsl.ResourceLimits.LevelLimit): Bounds =
    (Some(0d), Some(limit.max.toDouble), Some(limit.warnAt.toDouble))

  private val limitsBySubtype: Map[(String, String), (String, Map[String, Bounds])] = Map(
    ("Sponge", "level") -> ("spongeType", Map(
      "VolumeFilling"    -> meshSpongeBounds,
      "SurfaceUnfolding" -> meshSpongeBounds,
      "CubeSponge"       -> meshSpongeBounds,
      "RecursiveIAS"     -> (
        Some(menger.dsl.ResourceLimits.recursiveIasMinLevel.toDouble),
        Some(menger.dsl.ResourceLimits.recursiveIasMaxLevel.toDouble),
        None
      )
    )),
    ("TesseractSponge", "level") -> ("spongeType", Map(
      "VolumeRemoving" -> tesseractSpongeBounds(
        menger.dsl.ResourceLimits.tesseractSpongeVolumeLevel
      ),
      "SurfaceSubdividing" -> tesseractSpongeBounds(
        menger.dsl.ResourceLimits.tesseractSpongeSurfaceLevel
      )
    ))
  )

  def limitsBySubtypeOf(
    typeName: String,
    fieldName: String
  ): Option[(String, Map[String, Bounds])] =
    limitsBySubtype.get((typeName, fieldName))

  def limitsOf(typeName: String, fieldName: String): (Option[Double], Option[Double]) =
    fieldLimits.getOrElse((typeName, fieldName), (None, None))

  /** Level at or above which rendering gets slow (`ResourceLimits.*.warnAt`) -- the agent should
    * ask before going there (usability review 2026-09, session 2, F36). `TesseractSponge` takes
    * the smaller of its two variants' thresholds, like its `max` above. */
  private val fieldWarnLevels: Map[(String, String), Double] = Map(
    ("Sponge", "level") -> menger.dsl.ResourceLimits.cubeSpongeLevel.warnAt.toDouble,
    ("TesseractSponge", "level") -> math.min(
      menger.dsl.ResourceLimits.tesseractSpongeVolumeLevel.warnAt,
      menger.dsl.ResourceLimits.tesseractSpongeSurfaceLevel.warnAt
    ).toDouble
  )

  def warnAtOf(typeName: String, fieldName: String): Option[Double] =
    fieldWarnLevels.get((typeName, fieldName))
