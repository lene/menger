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
      "explicit `color` replaces its material's colour INCLUDING alpha, so an opaque `color` " +
      "turns glass or film opaque -- omit `color` or give it a low alpha to keep transparency.",
    "Emission makes a surface self-lit: flat, unshaded colour. There is no bloom, halo or " +
      "glow around objects, and no emission that falls off with distance.",
    "Animation: `def scene(t: Float): Scene` plus `val duration = <seconds>f` in the same " +
      "object; t is seconds in [0, duration]. The window plays it in real time, looping, and " +
      "the validator checks the scene at t = 0 and at t = duration.",
    "4D objects are rotated in 4D (`projection`), projected to 3D, then placed at `pos`. The " +
      "volume-filling TesseractSponge currently ignores `size`."
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
    ("TesseractSponge", "size") -> "Currently ignored for VolumeRemoving (known issue).",
    ("Sphere", "size") -> "Radius in world units.",
    ("Sponge", "color") -> "Replaces the material colour including alpha (see conventions).",
    ("TesseractSponge", "color") -> "Replaces the material colour including alpha.",
    ("Tesseract", "color") -> "Replaces the material colour including alpha.",
    ("Sphere", "color") -> "Replaces the material colour including alpha.",
    ("Sponge", "rotation") -> "Euler angles in radians around x, y, z.",
    ("Tesseract", "rotation") -> ("3D Euler angles in radians, applied after projection; " +
      "4D rotations go in `projection`."),
    ("TesseractSponge", "rotation") -> ("3D Euler angles in radians, applied after " +
      "projection; 4D rotations go in `projection`."),
    ("Tesseract", "edgeRadius") -> "Draws the edges as tubes of this radius.",
    ("TesseractSponge", "edgeRadius") -> "Draws the edges as tubes of this radius."
  )

  def descriptionOf(typeName: String, fieldName: String): Option[String] =
    fieldDescriptions.get((typeName, fieldName))
