package menger.objects

object LSystemPresets:
  /** Returns (axiom, rules, angle, segmentLength, initialWidth, decay, defaultIterations) */
  def apply(name: String): (String, Map[Char, String], Float, Float, Float, Float, Int) =
    val lowerName = name.toLowerCase
    require(presets.contains(lowerName), s"Unknown L-system preset: $name. " +
      s"Valid presets: ${presets.keys.mkString(", ")}")

    val (axiom, rules, angle, segLen, initWidth, decay, defaultIters) =
      presets(lowerName)
    (axiom, rules, angle, segLen, initWidth, decay, defaultIters)

  def exists(name: String): Boolean = presets.contains(name.toLowerCase)

  def names: Iterable[String] = presets.keys

  private val presets: Map[String, (String, Map[Char, String], Float, Float, Float, Float, Int)] = Map(
    "tree" -> ("F", Map('F' -> "F[+F]F[-F]F"), 25.7f, 0.3f, 0.08f, 0.7f, 4),
    "bush" -> ("F", Map('F' -> "FF+[+F-F-F]-[-F+F+F]"), 22.5f, 0.15f, 0.05f, 0.8f, 3),
    "fern3d" -> ("F", Map('F' -> "F[&F]F[^F][&F]"), 30.0f, 0.2f, 0.06f, 0.75f, 3),
    "hilbert3d" -> ("X", Map(
      'X' -> "^<XF^<XFX-F^>>XFX&F+>>XFX-F>X->",
      'F' -> "F"
    ), 90.0f, 0.1f, 0.05f, 1.0f, 4),
    "kochisland" -> ("F+F+F+F",
      Map('F' -> "F+f-FF+F+FF+Ff+FF-f+FF-F-FF-Ff-FFF"),
      90.0f, 0.05f, 0.03f, 1.0f, 2),
    // 4D presets — use > and < for ana/kata-axis rotations
    "hilbert4d" -> ("X", Map(
      'X' -> "^<XF^<XFX-F^>>XFX&F+>>XFX-F>X->",
      'F' -> "F"
    ), 90.0f, 0.15f, 0.05f, 0.7f, 4),
    "tree4d" -> ("F", Map(
      'F' -> "F[>+F]F[<-F]F"
    ), 25.7f, 0.3f, 0.08f, 0.7f, 4)
  )
