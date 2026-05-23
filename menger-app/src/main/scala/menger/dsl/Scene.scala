package menger.dsl

import menger.config.CameraConfig
import menger.config.SceneConfig

/** Complete scene definition with camera, objects, and lighting.
  *
  * Objects can be specified in two ways (mutually usable, root takes precedence):
  *   - `objects`: flat list for simple scenes (backward-compatible)
  *   - `root`: scene graph node enabling transform hierarchy and material inheritance
  *
  * @param camera Camera configuration (position, lookAt, up)
  * @param objects Flat list of scene objects (spheres, cubes, sponges, etc.)
  * @param lights List of lights (directional, point)
  * @param planes Floor/wall planes (up to 4 simultaneous planes)
  * @param caustics Optional caustics configuration for photon mapping
  * @param root Optional scene graph root node (overrides objects when set)
  * @param render Optional render settings (quality, shadows, ray depth, etc.)
  */
case class Fog(density: Float, color: Color = Color(0.8f, 0.8f, 0.9f))

case class Scene(
  camera: Camera = Camera.Default,
  objects: List[SceneObject] = List.empty,
  lights: List[Light] = List.empty,
  planes: List[Plane] = List.empty,
  caustics: Option[Caustics] = None,
  background: Option[Color] = None,
  fog: Option[Fog] = None,
  envMap: Option[String] = None,
  root: Option[SceneNode] = None,
  render: Option[RenderSettings] = None
):
  require(objects.nonEmpty || root.isDefined, "Scene must contain at least one object or a root node")

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
    new Scene(camera = camera, objects = List(obj))

  /** Create a scene with a single object and lights */
  def apply(camera: Camera, obj: SceneObject, lights: List[Light]): Scene =
    new Scene(camera = camera, objects = List(obj), lights = lights)

  /** Create a scene with objects, lights, and caustics */
  def apply(camera: Camera, objects: List[SceneObject], lights: List[Light], caustics: Caustics): Scene =
    new Scene(camera = camera, objects = objects, lights = lights, caustics = Some(caustics))

  /** Create a scene from a scene graph root node */
  def apply(camera: Camera, root: SceneNode): Scene =
    new Scene(camera = camera, root = Some(root))

  /** Create a scene from a scene graph root node with lights */
  def apply(camera: Camera, root: SceneNode, lights: List[Light]): Scene =
    new Scene(camera = camera, lights = lights, root = Some(root))
