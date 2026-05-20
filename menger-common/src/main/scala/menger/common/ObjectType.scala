package menger.common

object ObjectType:

  val VALID_TYPES: Set[String] = Set(
    "sphere",
    "cone",
    "plane",
    "cube",
    "parametric",
    "tetrahedron",
    "octahedron",
    "dodecahedron",
    "icosahedron",
    "sponge-volume",
    "sponge-surface",
    "cube-sponge",
    "sponge-recursive-ias",
    "tesseract",
    "pentachoron",
    "16-cell",
    "24-cell",
    "120-cell",
    "600-cell",
    "tesseract-sponge-volume",
    "tesseract-sponge-surface",
    "menger4d",
    "sierpinski4d"
  )

  val MENGER4D_TYPES: Set[String] = Set("menger4d")

  val SIERPINSKI4D_TYPES: Set[String] = Set("sierpinski4d")

  val SPONGE_TYPES: Set[String] = Set(
    "sponge-volume",
    "sponge-surface",
    "cube-sponge",
    "sponge-recursive-ias"
  )

  val PROJECTED_4D_TYPES: Set[String] = Set(
    "tesseract",
    "pentachoron",
    "16-cell",
    "24-cell",
    "120-cell",
    "600-cell",
    "tesseract-sponge-volume",
    "tesseract-sponge-surface"
  )

  val ANALYTICAL_PRIMITIVE_TYPES: Set[String] = Set("sphere", "cone", "plane")

  val TRIANGLE_MESH_TYPES: Set[String] =
    Set("cube", "parametric", "tetrahedron", "octahedron", "dodecahedron", "icosahedron") ++
    SPONGE_TYPES.filter(_ != "cube-sponge") ++
    PROJECTED_4D_TYPES

  // Deprecated names map to new canonical names
  private val DEPRECATED_ALIASES: Map[String, String] = Map(
    "sponge" -> "sponge-volume",
    "sponge-2" -> "sponge-surface",
    "tesseract-sponge" -> "tesseract-sponge-volume",
    "tesseract-sponge-2" -> "tesseract-sponge-surface"
  )

  /** Normalize a type name by mapping deprecated aliases to canonical names */
  def normalize(objectType: String): String =
    val lower = objectType.toLowerCase
    DEPRECATED_ALIASES.getOrElse(lower, lower)

  /** Check if a type name is deprecated */
  def isDeprecated(objectType: String): Boolean =
    DEPRECATED_ALIASES.contains(objectType.toLowerCase)

  def isValid(objectType: String): Boolean =
    val normalized = normalize(objectType)
    VALID_TYPES.contains(normalized)

  def isSponge(objectType: String): Boolean =
    val normalized = normalize(objectType)
    SPONGE_TYPES.contains(normalized)

  def isProjected4D(objectType: String): Boolean =
    val normalized = normalize(objectType)
    PROJECTED_4D_TYPES.contains(normalized)

  def isSpongeOrCube(objectType: String): Boolean =
    val normalized = normalize(objectType)
    isSponge(objectType) || normalized == "cube"

  def is4DSponge(objectType: String): Boolean =
    val normalized = normalize(objectType)
    normalized == "tesseract-sponge-volume" || normalized == "tesseract-sponge-surface" ||
    MENGER4D_TYPES.contains(normalized) || SIERPINSKI4D_TYPES.contains(normalized)

  def isMenger4D(objectType: String): Boolean =
    MENGER4D_TYPES.contains(normalize(objectType))

  def isSierpinski4D(objectType: String): Boolean =
    SIERPINSKI4D_TYPES.contains(normalize(objectType))

  def isRecursiveIASSponge(objectType: String): Boolean =
    normalize(objectType) == "sponge-recursive-ias"

  def isAnalyticalPrimitive(objectType: String): Boolean =
    ANALYTICAL_PRIMITIVE_TYPES.contains(normalize(objectType))

  def isTriangleMesh(objectType: String): Boolean =
    TRIANGLE_MESH_TYPES.contains(normalize(objectType))

  def validTypesString: String =
    VALID_TYPES.toSeq.sorted.mkString(", ")

