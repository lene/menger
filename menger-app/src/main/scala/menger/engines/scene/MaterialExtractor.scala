package menger.engines.scene

import menger.ObjectSpec
import menger.common.Color
import menger.optix.Material

/**
 * Utility for extracting Material from ObjectSpec.
 *
 * Provides consistent material resolution logic:
 * - Use spec.material if provided (Material preset)
 * - Otherwise fallback to Material(color, ior) from spec fields
 * - Use default color if spec.color is not provided
 */
object MaterialExtractor:
  private val defaultColor = Color(0.7f, 0.7f, 0.7f)

  /**
   * Extract Material from ObjectSpec using the following precedence:
   * 1. spec.material (if provided)
   * 2. Material(spec.color, spec.ior) (if color provided)
   * 3. Material(defaultColor, spec.ior) (fallback)
   */
  def extract(spec: ObjectSpec): Material =
    spec.material match
      case Some(mat) => mat
      case None => Material(spec.color.getOrElse(defaultColor), spec.ior)
