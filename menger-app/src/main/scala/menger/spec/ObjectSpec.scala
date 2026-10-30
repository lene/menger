package menger.spec

import menger.common.Color
import menger.common.Material
import menger.common.TriangleMeshData
import menger.video.VideoTexture

/**
 * Object specification, as parsed from the CLI --object flag (menger.cli.ObjectSpecParser) or
 * built from a DSL scene object.
 * Format: type=TYPE:key=value:key=value...
 *
 * Success examples:
 *   type=sphere:pos=0,0,0:size=1.0:color=#FF0000:ior=1.5
 *   type=cube:pos=2,0,0:size=1.5:color=#0000FF
 *   type=sponge-volume:pos=-2,0,0:size=2.0:level=3:color=#00FF00
 *   type=sponge-surface:pos=0,2,0:size=2.0:level=2.5:color=#FFFF00
 *   type=cube-sponge:pos=1,0,0:size=2.0:level=2:color=#FF00FF
 *   type=sphere:pos=0,0,0:material=glass:ior=1.7
 *   type=sphere:pos=0,0,0:material=glass:ior=1.5:dispersion=33
 *   type=cube:pos=0,0,0:material=metal:color=#FFD700
 *   type=cube:pos=0,0,0:texture=brick.png
 *   type=tesseract:pos=0,0,0:size=2:rot-xw=30:rot-yw=15:color=#4488FF
 *
 * Error examples (input → error message):
 *   "type=invalid" → "Invalid object type: 'invalid'. Valid types: sphere, cube, ..."
 *   "pos=1,2,3" (missing type) → "Missing required 'type' field..."
 *   "type=sphere:pos=1,2" → "Invalid position format: '1,2'. Expected format: pos=x,y,z..."
 *   "type=sphere:size=abc" → "Invalid size value 'abc': For input string: \"abc\"..."
 *   "type=sponge-volume:pos=0,0,0" → "Sponge object requires 'level' field..."
 *   "type=sphere:material=unknown" → "Unknown material preset: 'unknown'. Valid presets: ..."
 *   "type=sphere:color=notahex" → "Invalid color value 'notahex': For input string: \"notahex\"..."
 *   "type=tesseract:eye-w=1:screen-w=2" → "eye-w must be greater than screen-w..."
 */
case class ObjectSpec(
  objectType: String,
  x: Float = 0.0f,
  y: Float = 0.0f,
  z: Float = 0.0f,
  size: Float = 1.0f,
  level: Option[Float] = None,
  color: Option[Color] = None,
  ior: Float = 1.0f,
  material: Option[Material] = None,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  projection4D: Option[Projection4DSpec] = None,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: ObjectRotation = ObjectRotation(),
  cone: ConeGeometry = ConeGeometry(),
  plane: PlaneGeometry = PlaneGeometry(),
  procedural: ProceduralSpec = ProceduralSpec(),
  textureMaps: TextureMaps = TextureMaps(),
  metallicMap: Option[String] = None,
  aoMap: Option[String] = None,
  heightMap: Option[String] = None,
  textureSet: Option[String] = None,
  textureSetRes: Option[String] = None,
  uvScale: Option[Float] = None,
  meshData: Option[TriangleMeshData] = None,
  distanceThreshold: Option[Int] = None,
  curveData: Option[CurveData] = None,
  lsystemPreset: Option[String] = None,
  lsystemAngle: Option[Float] = None,
  lsystemSeed: Option[Long] = None,
  lsystemDim: Int = 3,
  dispersion: Float = 0.0f
):
  require(texture.isEmpty || videoTexture.isEmpty, "texture and videoTexture are mutually exclusive")

  def hasEdgeRendering: Boolean = edgeRadius.isDefined || edgeMaterial.isDefined
  def rotX: Float = rotation.x
  def rotY: Float = rotation.y
  def rotZ: Float = rotation.z
  def proceduralType: Int = procedural.proceduralType
  def proceduralScale: Float = procedural.scale
  def imageTextureKey: Option[String] = texture.orElse(videoTexture.map(_.textureKey))
  def normalMap: Option[String] = textureMaps.normalMap
  def roughnessMap: Option[String] = textureMaps.roughnessMap
  def apex: Option[(Float, Float, Float)] = cone.apex
  def base: Option[(Float, Float, Float)] = cone.base
  def radius: Option[Float] = cone.radius
  def normal: Option[(Float, Float, Float)] = plane.normal
  def distance: Option[Float] = plane.distance
  def color2: Option[Color] = plane.color2
  def checkerSize: Float = plane.checkerSize

  /** The 4D w-scale; 1 without a projection. */
  def wScale: Float = projection4D.fold(Projection4DSpec.DefaultWScale)(_.wScale)

  /** This spec without its 4D view (eye, screen, rotations) but with its w-scale: a view
    * change re-projects the same 4D mesh, a w-scale change is a different mesh (menger#65). */
  def withoutView: ObjectSpec =
    copy(projection4D =
      Some(wScale).filter(_ != Projection4DSpec.DefaultWScale).map(w => Projection4DSpec(wScale = w))
    )

object ObjectSpec:

  /** The 3D types with edge tubes: triangle meshes whose edges come from their faces
    * (MeshEdges3D, menger#132). A recursive-IAS sponge has no flat mesh to take them from. */
  val Edge3DTypes: Set[String] = Set(
    "cube", "sponge-volume", "sponge-surface",
    "tetrahedron", "octahedron", "dodecahedron", "icosahedron"
  )

  def hasEdgeTubes(objType: String): Boolean =
    ObjectType.isProjected4D(objType) || Edge3DTypes.contains(ObjectType.normalize(objType))
