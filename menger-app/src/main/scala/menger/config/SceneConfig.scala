package menger.config

import menger.ObjectSpec

/**
 * Scene configuration specifying what geometry to render using object specifications.
 */
case class SceneConfig(
  // Modern multi-object mode (only supported mode)
  objectSpecs: Option[List[ObjectSpec]] = None
):
  /** Returns true if using multi-object mode */
  def isMultiObject: Boolean = objectSpecs.isDefined

  /** Returns true if no objects configured */
  def isEmpty: Boolean = objectSpecs.isEmpty || objectSpecs.get.isEmpty

object SceneConfig:
  /**
   * Default configuration: no objects (will need to be configured)
   */
  val Default: SceneConfig = SceneConfig()

  /**
   * Create scene config for multiple objects.
   */
  def multiObject(specs: List[ObjectSpec]): SceneConfig =
    SceneConfig(objectSpecs = Some(specs))
