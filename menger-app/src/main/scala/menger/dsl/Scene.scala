package menger.dsl

import menger.config.CameraConfig
import menger.config.SceneConfig

/** Complete scene definition with camera, objects, and lighting.
  *
  * @param camera Camera configuration (position, lookAt, up)
  * @param objects List of scene objects (spheres, cubes, sponges, etc.)
  * @param lights List of lights (directional, point)
  * @param plane Optional floor/wall plane
  * @param caustics Optional caustics configuration for photon mapping
  */
case class Scene(
  camera: Camera = Camera.Default,
  objects: List[SceneObject] = List.empty,
  lights: List[Light] = List.empty,
  plane: Option[Plane] = None,
  caustics: Option[Caustics] = None
):
  require(objects.nonEmpty, "Scene must contain at least one object")

  /** Convert scene to SceneConfig for rendering */
  def toSceneConfig: SceneConfig =
    val objectSpecs = objects.map(_.toObjectSpec)
    SceneConfig.multiObject(objectSpecs)

  /** Convert camera to CameraConfig for rendering */
  def toCameraConfig: CameraConfig =
    camera.toCameraConfig

object Scene:
  /** Create a scene with a single object */
  def apply(camera: Camera, obj: SceneObject): Scene =
    new Scene(camera, List(obj), List.empty, None, None)

  /** Create a scene with a single object and lights */
  def apply(camera: Camera, obj: SceneObject, lights: List[Light]): Scene =
    new Scene(camera, List(obj), lights, None, None)

  /** Create a scene with objects, lights, and caustics */
  def apply(camera: Camera, objects: List[SceneObject], lights: List[Light], caustics: Caustics): Scene =
    new Scene(camera, objects, lights, None, Some(caustics))
