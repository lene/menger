package menger.common

/**
 * Object type definitions and validation for geometric objects.
 *
 * Provides centralized validation and categorization of object types
 * to ensure consistency across the application.
 */
object ObjectType:

  /**
   * All valid object types supported by the renderer.
   */
  val VALID_TYPES: Set[String] = Set(
    "sphere",
    "cube",
    "sponge-volume",
    "sponge-surface",
    "cube-sponge"
  )

  /**
   * Object types that represent Menger sponge variants.
   */
  val SPONGE_TYPES: Set[String] = Set(
    "sponge-volume",
    "sponge-surface",
    "cube-sponge"
  )

  /**
   * Legacy object types (for backward compatibility with single-object mode).
   * Excludes cube-sponge which is only available in multi-object mode.
   */
  val LEGACY_TYPES: Set[String] = Set(
    "sphere",
    "cube",
    "sponge-volume",
    "sponge-surface"
  )

  /**
   * Checks if the given object type is valid.
   *
   * @param objectType Object type string (case-insensitive)
   * @return true if the type is valid, false otherwise
   */
  def isValid(objectType: String): Boolean =
    VALID_TYPES.contains(objectType.toLowerCase)

  /**
   * Checks if the given object type is a legacy type.
   *
   * @param objectType Object type string (case-insensitive)
   * @return true if the type is a legacy type, false otherwise
   */
  def isLegacy(objectType: String): Boolean =
    LEGACY_TYPES.contains(objectType.toLowerCase)

  /**
   * Checks if the given object type is a Menger sponge variant.
   *
   * @param objectType Object type string (case-insensitive)
   * @return true if the type is a sponge variant, false otherwise
   */
  def isSponge(objectType: String): Boolean =
    SPONGE_TYPES.contains(objectType.toLowerCase)

  /**
   * Checks if the given object type is a sponge variant or cube.
   *
   * @param objectType Object type string (case-insensitive)
   * @return true if the type is a sponge variant or cube, false otherwise
   */
  def isSpongeOrCube(objectType: String): Boolean =
    isSponge(objectType) || objectType.toLowerCase == "cube"

  /**
   * Formats a comma-separated list of valid types for error messages.
   *
   * @return Human-readable list of valid types
   */
  def validTypesString: String =
    VALID_TYPES.toSeq.sorted.mkString(", ")

  /**
   * Formats a comma-separated list of legacy types for error messages.
   *
   * @return Human-readable list of legacy types
   */
  def legacyTypesString: String =
    LEGACY_TYPES.toSeq.sorted.mkString(", ")
