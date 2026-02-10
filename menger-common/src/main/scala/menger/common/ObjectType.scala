package menger.common

object ObjectType:

  val VALID_TYPES: Set[String] = Set(
    "sphere",
    "cube",
    "sponge-volume",
    "sponge-surface",
    "cube-sponge",
    "tesseract",
    "tesseract-sponge-volume",
    "tesseract-sponge-surface"
  )

  val SPONGE_TYPES: Set[String] = Set(
    "sponge-volume",
    "sponge-surface",
    "cube-sponge"
  )

  val PROJECTED_4D_TYPES: Set[String] = Set(
    "tesseract",
    "tesseract-sponge-volume",
    "tesseract-sponge-surface"
  )

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
    normalized == "tesseract-sponge-volume" || normalized == "tesseract-sponge-surface"

  def validTypesString: String =
    VALID_TYPES.toSeq.sorted.mkString(", ")

