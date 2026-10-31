package menger.tools

import scala.util.Try

import menger.common.CausticsConfig
import menger.common.Color
import menger.common.Light
import menger.common.Material
import menger.common.Vector
import menger.common.x
import menger.common.y
import menger.common.z
import menger.config.PlaneConfig
import menger.dsl.Scene
import menger.dsl.SceneObject
import menger.engines.SceneConverter
import menger.spec.ObjectSpec
import upickle.default.ReadWriter

/** The evaluated values of a scene at one time, as the scene agent needs them to describe and
  * check a turn (usability review 2026-10, session 4, F90/F89/F102): the agent only parses
  * scene text, so it could not resolve local vals, constants, `if`/`else` legs or `math.cos`.
  * Built from `SceneConverter.convert`'s resolved `ObjectSpec`s, the same values the renderer
  * gets. Vectors are `[x, y, z]`, colours `[r, g, b, a]`. */
object SceneSnapshot:

  case class MaterialValues(
    color: List[Float],
    ior: Float,
    roughness: Float,
    metallic: Float,
    specular: Float,
    emission: Float,
    filmThickness: Float,
    dispersion: Float
  ) derives ReadWriter

  case class ProjectionValues(
    eyeW: Float, screenW: Float, rotXW: Float, rotYW: Float, rotZW: Float, wScale: Float
  ) derives ReadWriter

  /** `objectType` is the renderer's name, which carries the subtype (`sponge-volume`,
    * `sponge-surface`); `dslType` is the DSL case class the scene wrote (`Sponge`), the name
    * the agent shows (review 2026-10-14, menger#92). */
  case class ObjectValues(
    objectType: String,
    dslType: String,
    pos: List[Float],
    size: Float,
    level: Option[Float],
    rotation: List[Float],
    color: Option[List[Float]],
    material: Option[MaterialValues],
    edgeRadius: Option[Float],
    edgeMaterial: Option[MaterialValues],
    projection: Option[ProjectionValues],
    proceduralType: Int,
    proceduralScale: Float
  ) derives ReadWriter

  case class LightValues(
    kind: String,
    position: Option[List[Float]],
    direction: Option[List[Float]],
    color: List[Float],
    intensity: Float
  ) derives ReadWriter

  case class CameraValues(position: List[Float], lookAt: List[Float], up: List[Float])
    derives ReadWriter

  /** A plane: `axis` X/Y/Z, `positive` = the side it faces, `value` its position on the axis;
    * `checkerColor` only for a checkered plane. Planes were missing, so a plane change was
    * missing from the agent's change list and its readback guessed (menger#92). */
  case class PlaneValues(
    axis: String,
    positive: Boolean,
    value: Float,
    color: Option[List[Float]],
    checkerColor: Option[List[Float]],
    material: Option[MaterialValues]
  ) derives ReadWriter

  // ponytail: caustics and render settings as their DSL case-class text -- enough for the agent
  // to show a change; model them field by field when a check needs one of them.
  case class Snapshot(
    t: Float,
    camera: CameraValues,
    objects: List[ObjectValues],
    lights: List[LightValues],
    background: Option[List[Float]],
    caustics: Option[String],
    render: Option[String],
    planes: List[PlaneValues] = Nil
  ) derives ReadWriter

  case class CameraSample(t: Float, position: List[Float], lookAt: List[Float])
    derives ReadWriter

  /** Spacing of `cameraPath` samples: fine enough to see a jump between two legs of a flight
    * (8b had three), cheap because only `scene(t).camera` is read. */
  val CameraPathStepSeconds = 0.05f
  val MaxCameraPathSamples = 401

  /** `None` when the scene does not convert (its failure is reported by the other checks). */
  def of(t: Float, scene: Scene): Option[Snapshot] =
    Try(SceneConverter.convert(scene, CausticsConfig.Disabled)).toOption.map { configs =>
      Snapshot(
        t = t,
        camera = CameraValues(
          vec(configs.camera.position), vec(configs.camera.lookAt), vec(configs.camera.up)
        ),
        objects = configs.scene.objectSpecs.getOrElse(Nil).zip(dslObjects(scene)).map(objectValues),
        lights = configs.lights.map(lightValues),
        background = configs.background.map(rgba),
        caustics = scene.caustics.map(_.toString),
        render = scene.render.map(_.toString),
        planes = configs.planes.map(planeValues)
      )
    }

  /** The scene's objects in the order `SceneConverter.flattenNode` turns them into specs (one
    * spec per object): `allLeafGeometry` walks the tree the same way. */
  private[tools] def dslObjects(scene: Scene): List[SceneObject] =
    scene.root.map(_.allLeafGeometry).getOrElse(scene.objects)

  /** The camera every `CameraPathStepSeconds` over `[0, duration]` (at most
    * `MaxCameraPathSamples`, then evenly spread); a time whose scene throws is skipped. */
  def cameraPath(f: Float => Scene, duration: Float): List[CameraSample] =
    if duration <= 0f then Nil
    else
      val count = math.min(MaxCameraPathSamples, (duration / CameraPathStepSeconds).toInt + 1)
      List.tabulate(count)(i => duration * i / (count - 1)).flatMap { t =>
        Try(f(t).camera).toOption.map { c =>
          CameraSample(t, List(c.position.x, c.position.y, c.position.z),
            List(c.lookAt.x, c.lookAt.y, c.lookAt.z))
        }
      }

  private def vec(v: Vector[3]): List[Float] = List(v.x, v.y, v.z)

  private def rgba(c: Color): List[Float] = List(c.r, c.g, c.b, c.a)

  private def materialValues(m: Material): MaterialValues =
    MaterialValues(
      rgba(m.color), m.ior, m.roughness, m.metallic, m.specular, m.emission, m.filmThickness,
      m.dispersion
    )

  private def planeValues(plane: PlaneConfig): PlaneValues =
    PlaneValues(
      axis = plane.spec.axis.toString,
      positive = plane.spec.positive,
      value = plane.spec.value,
      color = plane.colorSpec.map(c => rgba(c.color1)),
      checkerColor = plane.colorSpec.flatMap(_.color2).map(rgba),
      material = plane.material.map(materialValues)
    )

  private def objectValues(spec: ObjectSpec, dsl: SceneObject): ObjectValues =
    ObjectValues(
      objectType = spec.objectType,
      dslType = dsl.getClass.getSimpleName,
      pos = List(spec.x, spec.y, spec.z),
      size = spec.size,
      level = spec.level,
      rotation = List(spec.rotation.x.value, spec.rotation.y.value, spec.rotation.z.value),
      color = spec.color.map(rgba),
      material = spec.material.map(materialValues),
      edgeRadius = spec.edgeRadius,
      edgeMaterial = spec.edgeMaterial.map(materialValues),
      projection = spec.projection4D.map(p =>
        ProjectionValues(p.eyeW, p.screenW, p.rotXW.value, p.rotYW.value, p.rotZW.value, p.wScale)
      ),
      proceduralType = spec.procedural.proceduralType,
      proceduralScale = spec.procedural.scale
    )

  private def lightValues(light: Light): LightValues = light match
    case d: Light.Directional =>
      LightValues("Directional", None, Some(vec(d.direction)), rgba(d.color), d.intensity)
    case p: Light.Point =>
      LightValues("Point", Some(vec(p.position)), None, rgba(p.color), p.intensity)
    case a: Light.Area =>
      LightValues("Area", Some(vec(a.position)), Some(vec(a.normal)), rgba(a.color), a.intensity)
