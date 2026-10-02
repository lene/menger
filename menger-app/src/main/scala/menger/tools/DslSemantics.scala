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
    "Units: positions and sizes are world units; `pos` is an object's centre; all angles " +
      "(`rotation`, 4D rotations) are radians.",
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
    "Emission makes a surface self-lit: flat, unshaded colour. There is no bloom, halo or " +
      "glow around objects, and no emission that falls off with distance.",
    "Animation: `def scene(t: Float): Scene` plus `val duration = <seconds>f` in the same " +
      "object; t is seconds in [0, duration]. The window plays it in real time, looping, and " +
      "the validator checks the scene at t = 0 and at t = duration. An animated scene has no " +
      "`SceneRegistry.register` line: register takes a static Scene only, so drop it when " +
      "turning `val scene` into `def scene(t: Float)` (usability review 2026-09, F46).",
    "4D objects are rotated in 4D (`projection`), projected to 3D, then placed at `pos`. " +
      "The regular 4D polytopes are Pentachoron (5-cell), Tesseract (8-cell), Hexadecachoron " +
      "(16-cell), Icositetrachoron (24-cell), Hecatonicosachoron (120-cell) and Hexacosichoron " +
      "(600-cell). Each 4D object has its own `projection`, so one object can be rotated in " +
      "4D next to an unrotated one; in the render window, Shift+drag adds the same 4D " +
      "rotation to every 4D object.",
    "Procedural colouring (`proceduralType`) is evaluated at the hit point's world position. " +
      "For a 4D object that is the projected 3D position, after `projection` and `pos`, so the " +
      "colours follow the projected shape and change when the 4D rotation changes " +
      "(usability review 2026-09, session 2, F43).",
    "Image textures and texture maps (`texture`, `videoTexture`, `normalMap`, " +
      "`roughnessMap`) need surface UV coordinates. 4D objects, edge tubes (`edgeRadius`), " +
      "curves and an L-system's branches have no UV coordinates, so these fields have no " +
      "effect on them; " +
      "use a material colour or a `proceduralType` there instead.",
    "Glass on a TesseractSponge from level 2 up renders as chaotic, fragmented refraction: the " +
      "projected sponge has thousands of overlapping refracting layers and a ray gets at most 5 " +
      "bounces. Prefer an opaque or metal material there, or level 1 for glass, and say so if " +
      "the user asks for glass (usability review 2026-09, session 2, F55).",
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
      "neighbouring integer levels. Cost grows ~20x per level."),
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
        "world position as RGB, mirrored at 0 on each axis; 9 heatmap REPLACES the colour " +
        "with a blue-to-red noise gradient. It belongs to the look of the material it " +
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
