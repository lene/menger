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
 *   type=cylinder:p0=0,-0.1,0:p1=0,0,0:radius=1.5:material=matte
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
  projection: Option[ProjectionLike] = None,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: ObjectRotation = ObjectRotation(),
  cone: ConeGeometry = ConeGeometry(),
  cylinder: CylinderGeometry = CylinderGeometry(),
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
  dispersion: Float = 0.0f,
  // The n-cube's dimension (menger#163); None for every other type.
  dimension: Option[Int] = None,
  // Per-axis geometry scale, applied after `size` and before rotation and projection (menger#164):
  // one factor >= 0 per axis of the object's own dimension. None means all 1.
  scale: Option[IndexedSeq[Float]] = None
):
  require(texture.isEmpty || videoTexture.isEmpty, "texture and videoTexture are mutually exclusive")
  scale.foreach { factors =>
    require(
      factors.forall(f => f.isFinite && f >= 0f),
      s"scale factors must be finite and >= 0, got ${factors.mkString(", ")}"
    )
  }

  def hasEdgeRendering: Boolean = edgeRadius.isDefined || edgeMaterial.isDefined
  def rotX: Float = rotation.x.value
  def rotY: Float = rotation.y.value
  def rotZ: Float = rotation.z.value
  def proceduralType: Int = procedural.proceduralType
  def proceduralScale: Float = procedural.scale
  def imageTextureKey: Option[String] = texture.orElse(videoTexture.map(_.textureKey))
  def normalMap: Option[String] = textureMaps.normalMap
  def roughnessMap: Option[String] = textureMaps.roughnessMap
  def apex: Option[(Float, Float, Float)] = cone.apex
  def base: Option[(Float, Float, Float)] = cone.base
  def radius: Option[Float] = cone.radius
  def p0: Option[(Float, Float, Float)] = cylinder.p0
  def p1: Option[(Float, Float, Float)] = cylinder.p1
  def cylinderRadius: Option[Float] = cylinder.radius
  def normal: Option[(Float, Float, Float)] = plane.normal
  def distance: Option[Float] = plane.distance
  def color2: Option[Color] = plane.color2
  def checkerSize: Float = plane.checkerSize

  /** The 4D shorthand view, which every 4D reader and the GPU paths understand; None for a
    * [[ProjectionSpec]] (see [[isLegacyView]]). */
  def projection4D: Option[Projection4DSpec] = projection.collect { case p: Projection4DSpec => p }

  /** The general nD view, which only the CPU path renders (menger#162). */
  def projectionN: Option[ProjectionSpec] = projection.collect { case p: ProjectionSpec => p }

  /** True unless the spec carries a [[ProjectionSpec]]: only a legacy view may take a 4D fast
    * path or be rotated by the window. */
  def isLegacyView: Boolean = projectionN.isEmpty

  /** The per-axis scale of an object with `dimension` axes: all 1 without a `scale`. */
  def axisScales(dimension: Int): IndexedSeq[Float] =
    require(
      scale.forall(_.size == dimension),
      s"scale has ${scale.fold(0)(_.size)} factors, but ${objectType} is $dimension-dimensional " +
        s"and needs $dimension"
    )
    scale.getOrElse(IndexedSeq.fill(dimension)(ObjectSpec.IdentityScale))

  /** This spec without its view (eye, screen, rotations; a [[ProjectionSpec]] altogether): a view
    * change re-projects the same mesh, while a `scale` change is a different mesh (menger#65,
    * #164), and stays. */
  def withoutView: ObjectSpec = copy(projection = None)

object ObjectSpec:

  val IdentityScale: Float = 1f

  /** The dimension of the 4D polytope and sponge types, whose `scale` has this many factors. */
  val LegacyDimension: Int = 4

  /** The 3D types with edge tubes: triangle meshes whose edges come from their faces
    * (MeshEdges3D, menger#132). A recursive-IAS sponge has no flat mesh to take them from. */
  val Edge3DTypes: Set[String] = Set(
    "cube", "sponge-volume", "sponge-surface",
    "tetrahedron", "octahedron", "dodecahedron", "icosahedron"
  )

  def hasEdgeTubes(objType: String): Boolean =
    ObjectType.isEdgeTubeMesh(objType) || Edge3DTypes.contains(ObjectType.normalize(objType))
