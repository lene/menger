package menger.common

object ObjectType:

  val VALID_TYPES: Set[String] = Set(
    "sphere",
    "cube",
    "sponge-volume",
    "sponge-surface",
    "cube-sponge",
    "tesseract"
  )

  val SPONGE_TYPES: Set[String] = Set(
    "sponge-volume",
    "sponge-surface",
    "cube-sponge"
  )

  val HYPERCUBE_TYPES: Set[String] = Set("tesseract")

  def isValid(objectType: String): Boolean =
    VALID_TYPES.contains(objectType.toLowerCase)

  def isSponge(objectType: String): Boolean =
    SPONGE_TYPES.contains(objectType.toLowerCase)

  def isHypercube(objectType: String): Boolean =
    HYPERCUBE_TYPES.contains(objectType.toLowerCase)

  def isSpongeOrCube(objectType: String): Boolean =
    isSponge(objectType) || objectType.toLowerCase == "cube"

  def validTypesString: String =
    VALID_TYPES.toSeq.sorted.mkString(", ")

