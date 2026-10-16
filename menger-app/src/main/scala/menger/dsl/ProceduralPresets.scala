package menger.dsl

/** The built-in procedural textures a scene can name with `proceduralType`, in one table
  * (review 2026-10-14, menger#81): the id/name list existed five times -- optix-jni's shader
  * enum, `ObjectSpec`'s CLI names, the manifest prose, the agent's regex over that prose and
  * its rules file. The CLI names, the manifest's `proceduralPresets` section and the field
  * description all come from here; a test holds the ids to optix-jni's `ProceduralType`. The
  * ids are written out because the DSL's compile view does not include optix-jni. */
object ProceduralPresets:

  /** What a preset does to the material's colour. */
  enum Effect(val wire: String):
    /** The material's own colour, varied in brightness by the pattern (same hue). */
    case Modulates extends Effect("modulates")
    /** The pattern's colour instead of the material's. */
    case Replaces  extends Effect("replaces")

  final case class Preset(id: Int, name: String, effect: Effect, note: String)

  val None: Preset = Preset(0, "none", Effect.Modulates, "no pattern (default)")

  val All: List[Preset] = List(
    None,
    Preset(1, "value_noise", Effect.Modulates, ""),
    Preset(2, "fbm", Effect.Modulates, ""),
    Preset(3, "worley", Effect.Modulates, ""),
    Preset(4, "gradient", Effect.Modulates, ""),
    Preset(5, "wood", Effect.Modulates, ""),
    Preset(6, "marble", Effect.Modulates, ""),
    Preset(7, "layered_noise", Effect.Modulates, ""),
    Preset(8, "xyz_rgb", Effect.Replaces,
      "(|x|, |y|, |z| mod 1) of the world position as RGB, mirrored at 0 on each axis"),
    Preset(9, "heatmap", Effect.Replaces, "a blue-to-red noise gradient"),
    Preset(10, "triplanar", Effect.Modulates, ""),
    Preset(11, "xyz_rgb_local", Effect.Replaces,
      "the position inside the object's own box, x, y, z each 0..1 -> R, G, B: every colour " +
        "once over the object at scale 1, following its position, size and rotation, no " +
        "mirroring -- use it for \"local\" or \"every colour once\" colouring, never move the " +
        "object for it (usability review 2026-10, session 3, F63)")
  )

  def byName(name: String): Option[Preset] = All.find(_.name == name.toLowerCase)

  /** The `proceduralType` field description, built from the table. */
  def description: String =
    def listed(effect: Effect) = All.filter(p => p.effect == effect && p != None)
      .map(p => s"${p.id} ${p.name}" + (if p.note.isEmpty then "" else s" (${p.note})"))
      .mkString(", ")
    "Built-in procedural texture, one of exactly these presets (structured list: the manifest's " +
      "`proceduralPresets`): 0 none (default). Modulating -- the material's own colour, varied " +
      s"in brightness by the pattern: ${listed(Effect.Modulates)}. Replacing -- the pattern's " +
      s"colour instead of the material's: ${listed(Effect.Replaces)}. A preset belongs to the " +
      "look of the material it imitates: swapping the material (e.g. wood to aluminium) drops a " +
      "pattern like wood."
