package menger.engines.scene

import menger.ObjectSpec
import menger.common.Color
import menger.common.Material

/**
 * Utility for extracting Material from ObjectSpec.
 *
 * Provides consistent material resolution logic:
 * - Use spec.material if provided (Material preset), tinted by spec.color if also given
 * - Otherwise fallback to Material(color, ior) from spec fields
 * - Use default color if spec.color is not provided
 */
object MaterialExtractor:
  private val defaultColor = Color(0.7f, 0.7f, 0.7f)

  /**
   * Extract Material from ObjectSpec using the following precedence:
   * 1. spec.material (if provided), with its color overridden by spec.color when both are
   *    set (Sprint 36 H5.3) -- material supplies the physical properties (roughness,
   *    metallic, ior, ...), color is a tint override on top, not a competing setting
   * 2. Material(spec.color, spec.ior) (if color provided, no material)
   * 3. Material(defaultColor, spec.ior) (fallback)
   */
  def extract(spec: ObjectSpec): Material =
    spec.material match
      case Some(mat) => spec.color.fold(mat)(c => mat.copy(color = c))
      case None => Material(spec.color.getOrElse(defaultColor), spec.ior, dispersion = spec.dispersion)
