package menger.config

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.math.Vector3
import menger.ObjectSpec

/**
 * Scene configuration specifying what geometry to render.
 *
 * Supports two modes:
 * 1. Single-object legacy mode: Use spongeType, level, color, etc.
 * 2. Multi-object mode: Use objectSpecs
 */
case class SceneConfig(
  // Legacy single-object parameters (deprecated, use objectSpecs instead)
  spongeType: String = "sphere",
  level: Float = 0f,
  lines: Boolean = false,
  color: Color = Color.WHITE,
  sphereRadius: Float = 0.5f,
  ior: Float = 1.5f,
  scale: Float = 1.0f,
  center: Vector3 = Vector3.Zero,

  // Modern multi-object mode
  objectSpecs: Option[List[ObjectSpec]] = None
):
  /** Returns true if using multi-object mode (new) vs single-object mode (legacy) */
  def isMultiObject: Boolean = objectSpecs.isDefined

  /** Returns true if using legacy single-object mode */
  def isSingleObject: Boolean = !isMultiObject

object SceneConfig:
  /**
   * Default configuration: single sphere at origin
   */
  val Default: SceneConfig = SceneConfig()

  /**
   * Create scene config for single object (legacy mode).
   */
  def singleObject(
    objectType: String,
    level: Float = 0f,
    lines: Boolean = false,
    color: Color = Color.WHITE,
    sphereRadius: Float = 0.5f,
    ior: Float = 1.5f,
    scale: Float = 1.0f,
    center: Vector3 = Vector3.Zero
  ): SceneConfig =
    SceneConfig(
      spongeType = objectType,
      level = level,
      lines = lines,
      color = color,
      sphereRadius = sphereRadius,
      ior = ior,
      scale = scale,
      center = center,
      objectSpecs = None
    )

  /**
   * Create scene config for multiple objects (new mode).
   */
  def multiObject(specs: List[ObjectSpec]): SceneConfig =
    SceneConfig(objectSpecs = Some(specs))
