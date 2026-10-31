package menger.dsl

import scala.annotation.targetName

import menger.objects.LSystemPresets
import menger.spec.ConeGeometry
import menger.spec.CurveData
import menger.spec.Degrees
import menger.spec.ObjectRotation
import menger.spec.ObjectSpec
import menger.spec.ObjectType
import menger.spec.ProceduralSpec
import menger.spec.Projection4DSpec
import menger.spec.Radians
import menger.spec.TextureMaps
import menger.video.VideoTexture

/** Base trait for all scene objects */
sealed trait SceneObject:
  def pos: Vec3
  def size: Float
  def material: Option[Material]
  def color: Option[Color]
  def ior: Float
  def texture: Option[String]
  def videoTexture: Option[VideoTexture]
  def normalMap: Option[String]
  def roughnessMap: Option[String]
  def proceduralType: Int
  def proceduralScale: Float
  def rotation: Vec3
  def toObjectSpec: ObjectSpec
  def materialsToValidate: List[Material] = material.toList

  protected def baseObjectSpec(
    objectType: String,
    level: Option[Float] = None,
    projection4D: Option[Projection4DSpec] = None,
    edgeRadius: Option[Float] = None,
    edgeMaterial: Option[menger.common.Material] = None,
    meshData: Option[menger.common.TriangleMeshData] = None,
    curveData: Option[CurveData] = None
  ): ObjectSpec =
    ObjectSpec(
      objectType = objectType,
      x = pos.x,
      y = pos.y,
      z = pos.z,
      size = size,
      level = level,
      color = color.map(_.toCommonColor),
      ior = material.map(_.ior).getOrElse(ior),
      material = material.map(_.toCoreMaterial),
      texture = texture,
      videoTexture = videoTexture,
      projection4D = projection4D,
      edgeRadius = edgeRadius,
      edgeMaterial = edgeMaterial,
      rotation = ObjectRotation(Radians(rotation.x), Radians(rotation.y), Radians(rotation.z)),
      procedural = ProceduralSpec(proceduralType, proceduralScale),
      textureMaps = TextureMaps(normalMap = normalMap, roughnessMap = roughnessMap),
      meshData = meshData,
      curveData = curveData
    )

/** Sphere object */
case class Sphere(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec = baseObjectSpec("sphere")

object Sphere:
  // Material-only constructor (at origin)
  def apply(material: Material): Sphere =
    Sphere(pos = Vec3.Zero, material = Some(material))

  // Position + material
  def apply(pos: Vec3, material: Material): Sphere =
    Sphere(pos, Some(material))

  @targetName("spherePosMatSize")
  def apply(pos: Vec3, material: Material, size: Float): Sphere =
    Sphere(pos, Some(material), size = size)


/** Cube object */
case class Cube(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec =
    baseObjectSpec(
      "cube", edgeRadius = edgeRadius, edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
    )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList

object Cube:
  // Material-only constructor (at origin)
  def apply(material: Material): Cube =
    Cube(pos = Vec3.Zero, material = Some(material))

  // Position + material
  def apply(pos: Vec3, material: Material): Cube =
    Cube(pos, Some(material))

  @targetName("cubePosMatSize")
  def apply(pos: Vec3, material: Material, size: Float): Cube =
    Cube(pos, Some(material), size = size)


// The platonic solids were CLI-only types; every object type has a DSL class (2026-10).

/** Tetrahedron: 4 triangular faces. */
case class Tetrahedron(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec =
    baseObjectSpec(
      "tetrahedron", edgeRadius = edgeRadius, edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
    )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList

/** Octahedron: 8 triangular faces. */
case class Octahedron(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec =
    baseObjectSpec(
      "octahedron", edgeRadius = edgeRadius, edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
    )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList

/** Dodecahedron: 12 pentagonal faces. */
case class Dodecahedron(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec =
    baseObjectSpec(
      "dodecahedron", edgeRadius = edgeRadius, edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
    )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList

/** Icosahedron: 20 triangular faces. */
case class Icosahedron(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec =
    baseObjectSpec(
      "icosahedron", edgeRadius = edgeRadius, edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
    )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList


/** Cone. Without `apex`/`base` it is centred on `pos`, `size` high along y (turned by `rotation`),
  * with a base radius of `size / 2`; `apex` and `base` are world coordinates and override
  * `pos`, `size` and `rotation`. */
case class Cone(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  apex: Option[Vec3] = None,
  base: Option[Vec3] = None,
  radius: Option[Float] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")
  require(radius.forall(_ > 0f), s"Radius must be positive, got ${radius.getOrElse(0f)}")

  def toObjectSpec: ObjectSpec =
    baseObjectSpec("cone").copy(cone = ConeGeometry(
      apex.map(v => (v.x, v.y, v.z)), base.map(v => (v.x, v.y, v.z)), radius
    ))


/** Sponge type enumeration for DSL */
enum SpongeType(val objectTypeName: String):
  case VolumeFilling extends SpongeType("sponge-volume")
  case SurfaceUnfolding extends SpongeType("sponge-surface")
  case CubeSponge extends SpongeType("cube-sponge")
  case RecursiveIAS extends SpongeType("sponge-recursive-ias")

/** Menger sponge fractal object */
case class Sponge(
  spongeType: SpongeType,
  pos: Vec3 = Vec3.Zero,
  level: Float,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(level >= 0f, s"Level must be non-negative, got $level")
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")
  ResourceLimits.levelLimitByObjectType.get(ObjectType.normalize(spongeType.objectTypeName)).foreach { limit =>
    require(level <= limit.max,
      s"${spongeType.objectTypeName} level $level exceeds hard maximum ${limit.max}")
  }
  require(
    (edgeRadius.isEmpty && edgeMaterial.isEmpty) ||
      ObjectSpec.Edge3DTypes.contains(ObjectType.normalize(spongeType.objectTypeName)),
    "edge tubes (edgeRadius, edgeMaterial) exist on VolumeFilling and SurfaceUnfolding " +
      s"sponges, not on $spongeType (menger#132)"
  )

  def toObjectSpec: ObjectSpec =
    baseObjectSpec(
      ObjectType.normalize(spongeType.objectTypeName), level = Some(level),
      edgeRadius = edgeRadius, edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
    )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList

object Sponge:
  // Type + level (at origin, no material)
  def apply(spongeType: SpongeType, level: Float): Sponge =
    Sponge(spongeType, Vec3.Zero, level)

  // Type + level + material (at origin)
  @targetName("spongeTypeLevelMat")
  def apply(spongeType: SpongeType, level: Float, material: Material): Sponge =
    Sponge(spongeType, Vec3.Zero, level, Some(material))

  // Type + level + material + size (at origin)
  @targetName("spongeTypeLevelMatSize")
  def apply(spongeType: SpongeType, level: Float, material: Material, size: Float): Sponge =
    Sponge(spongeType, Vec3.Zero, level, Some(material), size = size)

  // Position + type + level
  @targetName("spongePosTypeLevel")
  def apply(pos: Vec3, spongeType: SpongeType, level: Float): Sponge =
    Sponge(spongeType, pos, level)

  // Position + type + level + material
  @targetName("spongePosTypeLevelMat")
  def apply(pos: Vec3, spongeType: SpongeType, level: Float, material: Material): Sponge =
    Sponge(spongeType, pos, level, Some(material))

  // Position + type + level + material + size
  @targetName("spongePosTypeLevelMatSize")
  def apply(pos: Vec3, spongeType: SpongeType, level: Float, material: Material, size: Float): Sponge =
    Sponge(spongeType, pos, level, Some(material), size = size)


// Export SpongeType values for convenient imports
export SpongeType.{VolumeFilling, SurfaceUnfolding, CubeSponge, RecursiveIAS}


/** Tesseract sponge type enumeration for DSL */
enum TesseractSpongeType(val objectTypeName: String):
  case VolumeRemoving extends TesseractSpongeType("tesseract-sponge-volume")
  case SurfaceSubdividing extends TesseractSpongeType("tesseract-sponge-surface")

/** Tesseract (4D hypercube) object */
case class Tesseract(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  projection: Option[Projection4DSpec] = None,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec =
    baseObjectSpec(
      "tesseract",
      projection4D = projection,
      edgeRadius = edgeRadius,
      edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
    )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList

object Tesseract:
  // Material-only constructor (at origin)
  def apply(material: Material): Tesseract =
    Tesseract(pos = Vec3.Zero, material = Some(material))

  // Position + material
  def apply(pos: Vec3, material: Material): Tesseract =
    Tesseract(pos, Some(material))

  @targetName("tesseractPosMatSize")
  def apply(pos: Vec3, material: Material, size: Float): Tesseract =
    Tesseract(pos, Some(material), size = size)

// The other regular 4D polytopes, with the same fields and 4D handling as Tesseract. The
// renderer supported them all along, but only the CLI could create them -- a DSL scene (and so
// the scene agent) could not (usability review 2026-09, F17). Deliberately separate case
// classes: an intermediate trait would hide them from the manifest's Mirror-derived object
// list.

/** Pentachoron (5-cell, {3,3,3}): 5 tetrahedral cells, the 4D simplex. */
case class Pentachoron(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  projection: Option[Projection4DSpec] = None,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec = baseObjectSpec(
    "pentachoron", projection4D = projection, edgeRadius = edgeRadius,
    edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
  )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList

/** Hexadecachoron (16-cell, {3,3,4}): 16 tetrahedral cells, the 4D cross-polytope. */
case class Hexadecachoron(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  projection: Option[Projection4DSpec] = None,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec = baseObjectSpec(
    "16-cell", projection4D = projection, edgeRadius = edgeRadius,
    edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
  )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList

/** Icositetrachoron (24-cell, {3,4,3}): 24 octahedral cells, self-dual. */
case class Icositetrachoron(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  projection: Option[Projection4DSpec] = None,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec = baseObjectSpec(
    "24-cell", projection4D = projection, edgeRadius = edgeRadius,
    edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
  )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList

/** Hexacosichoron (600-cell, {3,3,5}): 600 tetrahedral cells. */
case class Hexacosichoron(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  projection: Option[Projection4DSpec] = None,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec = baseObjectSpec(
    "600-cell", projection4D = projection, edgeRadius = edgeRadius,
    edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
  )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList

/** Hecatonicosachoron (120-cell, {5,3,3}): 120 dodecahedral cells. */
case class Hecatonicosachoron(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  projection: Option[Projection4DSpec] = None,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec = baseObjectSpec(
    "120-cell", projection4D = projection, edgeRadius = edgeRadius,
    edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
  )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList


/** Tesseract sponge fractal object (4D) */
case class TesseractSponge(
  spongeType: TesseractSpongeType,
  pos: Vec3 = Vec3.Zero,
  level: Float,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  projection: Option[Projection4DSpec] = None,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(level >= 0f, s"Level must be non-negative, got $level")
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")
  ResourceLimits.levelLimitByObjectType.get(ObjectType.normalize(spongeType.objectTypeName)).foreach { limit =>
    require(level <= limit.max,
      s"${spongeType.objectTypeName} level $level exceeds hard maximum ${limit.max}")
  }

  def toObjectSpec: ObjectSpec =
    baseObjectSpec(
      ObjectType.normalize(spongeType.objectTypeName),
      level = Some(level),
      projection4D = projection,
      edgeRadius = edgeRadius,
      edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
    )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList

object TesseractSponge:
  // Type + level (at origin, no material)
  def apply(spongeType: TesseractSpongeType, level: Float): TesseractSponge =
    TesseractSponge(spongeType, Vec3.Zero, level)

  // Type + level + material (at origin)
  @targetName("tesseractSpongeTypeLevelMat")
  def apply(spongeType: TesseractSpongeType, level: Float, material: Material): TesseractSponge =
    TesseractSponge(spongeType, Vec3.Zero, level, Some(material))

  // Type + level + material + size (at origin)
  @targetName("tesseractSpongeTypeLevelMatSize")
  def apply(spongeType: TesseractSpongeType, level: Float, material: Material, size: Float): TesseractSponge =
    TesseractSponge(spongeType, Vec3.Zero, level, Some(material), size = size)

  // Position + type + level
  @targetName("tesseractSpongePosTypeLevel")
  def apply(pos: Vec3, spongeType: TesseractSpongeType, level: Float): TesseractSponge =
    TesseractSponge(spongeType, pos, level)

  // Position + type + level + material
  @targetName("tesseractSpongePosTypeLevelMat")
  def apply(pos: Vec3, spongeType: TesseractSpongeType, level: Float, material: Material): TesseractSponge =
    TesseractSponge(spongeType, pos, level, Some(material))

  // Position + type + level + material + size
  @targetName("tesseractSpongePosTypeLevelMatSize")
  def apply(pos: Vec3, spongeType: TesseractSpongeType, level: Float, material: Material, size: Float): TesseractSponge =
    TesseractSponge(spongeType, pos, level, Some(material), size = size)


// Export TesseractSpongeType values for convenient imports
export TesseractSpongeType.{VolumeRemoving, SurfaceSubdividing}

/** 4D Sierpinski gasket of the pentachoron (5-cell), an IFS fractal; was `Sierpinski4D`. */
case class PentachoronGasket(
  pos: Vec3 = Vec3.Zero,
  level: Float,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  projection: Option[Projection4DSpec] = None,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(level >= 0f, s"Level must be non-negative, got $level")
  // The manifest already advertised this ceiling; the constructor did not enforce it (review
  // 2026-10-14, menger#106).
  FieldConstraints.check("PentachoronGasket", "level", level)
  require(size > 0f, s"Size must be positive, got $size")

  def toObjectSpec: ObjectSpec =
    baseObjectSpec(
      "sierpinski4d",
      level = Some(level),
      projection4D = projection,
      edgeRadius = edgeRadius,
      edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
    )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList

object PentachoronGasket:
  def apply(level: Float): PentachoronGasket = PentachoronGasket(Vec3.Zero, level)
  def apply(level: Float, material: Material): PentachoronGasket =
    PentachoronGasket(Vec3.Zero, level, Some(material))

// Scenes written before the rename keep compiling (and the manifest lists only the new name).
@deprecated("use PentachoronGasket", "2026-10")
type Sierpinski4D = PentachoronGasket
@deprecated("use PentachoronGasket", "2026-10")
val Sierpinski4D: PentachoronGasket.type = PentachoronGasket

/** 4D Menger sponge as an IFS fractal, traced on the GPU without stored geometry: the
  * TesseractSponge's shape, at a fraction of its memory. */
case class Menger4D(
  pos: Vec3 = Vec3.Zero,
  level: Float,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  projection: Option[Projection4DSpec] = None,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(level >= 0f, s"Level must be non-negative, got $level")
  FieldConstraints.check("Menger4D", "level", level)
  require(size > 0f, s"Size must be positive, got $size")

  def toObjectSpec: ObjectSpec =
    baseObjectSpec(
      "menger4d",
      level = Some(level),
      projection4D = projection,
      edgeRadius = edgeRadius,
      edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
    )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList

/** 4D Sierpinski gasket of the hexadecachoron (16-cell), an IFS fractal. */
case class HexadecachoronGasket(
  pos: Vec3 = Vec3.Zero,
  level: Float,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  projection: Option[Projection4DSpec] = None,
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(level >= 0f, s"Level must be non-negative, got $level")
  FieldConstraints.check("HexadecachoronGasket", "level", level)
  require(size > 0f, s"Size must be positive, got $size")

  def toObjectSpec: ObjectSpec =
    baseObjectSpec(
      "hexadecachoron4d",
      level = Some(level),
      projection4D = projection,
      edgeRadius = edgeRadius,
      edgeMaterial = edgeMaterial.map(_.toCoreMaterial)
    )

  override def materialsToValidate: List[Material] = material.toList ++ edgeMaterial.toList


/** Parametric surface defined by f(u,v) -> Vec3, tessellated into a triangle mesh.
  *
  * @param f        Surface function mapping (u,v) parameters to a 3D point
  * @param uRange   Parameter range for u (default: (0, 2π))
  * @param vRange   Parameter range for v (default: (0, π))
  * @param uSteps   Grid resolution in u direction (default: 64)
  * @param vSteps   Grid resolution in v direction (default: 32)
  * @param closedU  Whether to weld seam in u direction (first == last column)
  * @param closedV  Whether to weld seam in v direction (first == last row)
  * @param pos      Object position in world space
  * @param size     Uniform scale factor
  * @param ior      Index of refraction (for glass/transparent materials)
  * @param material Optional material override
  * @param color    Optional color override
  * @param texture  Optional texture name
  * @param rotation Rotation in radians around each axis
  */
case class ParametricSurface(
  f: (Float, Float) => Vec3,
  uRange: (Float, Float) = (0f, 2f * math.Pi.toFloat),
  vRange: (Float, Float) = (0f, math.Pi.toFloat),
  uSteps: Int = 64,
  vSteps: Int = 32,
  closedU: Boolean = false,
  closedV: Boolean = false,
  pos: Vec3 = Vec3.Zero,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  material: Option[Material] = None,
  color: Option[Color] = None,
  texture: Option[String] = None,
  videoTexture: Option[VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  FieldConstraints.check("ParametricSurface", "uSteps", uSteps)
  FieldConstraints.check("ParametricSurface", "vSteps", vSteps)
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")
  require(
    uSteps.toLong * vSteps.toLong <= ResourceLimits.parametricSurfaceMaxSamples,
    s"uSteps * vSteps ($uSteps x $vSteps = ${uSteps.toLong * vSteps.toLong}) exceeds hard " +
      s"maximum ${ResourceLimits.parametricSurfaceMaxSamples} grid cells"
  )

  def toObjectSpec: ObjectSpec =
    val tupleF: (Float, Float) => (Float, Float, Float) =
      (u, v) => { val p = f(u, v); (p.x, p.y, p.z) }
    val mesh = menger.objects.ParametricTessellator.tessellate(
      tupleF, uRange, vRange, uSteps, vSteps, closedU, closedV
    )
    baseObjectSpec("parametric", meshData = Some(mesh))

/** Round cubic B-spline tube rendered via the OptiX built-in curves primitive.
  *
  * Control points are in world space. `closed = true` wraps the spline by
  * appending the first 3 points so the curve joins smoothly at the seam.
  *
  * @param points   Sequence of at least 4 control points in world space
  * @param radius   Uniform tube radius (used when `radii` is None)
  * @param radii    Optional per-control-point radii (must match `points.size`)
  * @param closed   Whether to close the loop (appends 3 wrap points)
  * @param material Optional material override
  */
case class Curve(
  points: Seq[Vec3],
  radius: Float = 0.05f,
  radii: Option[Seq[Float]] = None,
  closed: Boolean = false,
  pos: Vec3 = Vec3.Zero,
  size: Float = 1.0f,
  material: Option[Material] = None,
  color: Option[Color] = None,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[menger.video.VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  require(points.size >= 4, s"Curve requires at least 4 control points, got ${points.size}")
  require(java.lang.Float.isFinite(radius) && radius > 0f, s"Curve radius must be finite and positive, got $radius")
  require(points.forall(p => java.lang.Float.isFinite(p.x) && java.lang.Float.isFinite(p.y) && java.lang.Float.isFinite(p.z)),
    "Curve control points must have finite coordinates (no NaN or Inf)")
  radii.foreach(r =>
    require(r.size == points.size,
      s"radii length (${r.size}) must equal points length (${points.size})")
    require(r.forall(v => java.lang.Float.isFinite(v) && v > 0f), "All per-point radii must be finite and positive")
  )

  def toObjectSpec: ObjectSpec =
    val allPoints = if closed then points ++ points.take(3) else points
    val flatPoints = Vector.from(allPoints.flatMap(p => Seq(p.x, p.y, p.z)))
    val effectiveWidths: Seq[Float] = radii.getOrElse(Seq.fill(points.size)(radius))
    val allWidths = if closed then effectiveWidths ++ effectiveWidths.take(3) else effectiveWidths
    val flatWidths = Vector.from(allWidths)
    baseObjectSpec("curve", curveData = Some(CurveData(flatPoints, flatWidths)))

object Curve:
  /** Closed ring of control points in the XZ plane. */
  def circle(
    radius: Float,
    tubeRadius: Float,
    segments: Int = 64,
    material: Option[Material] = None
  ): Curve =
    val pts = (0 until segments).map { i =>
      val angle = 2f * math.Pi.toFloat * i / segments
      Vec3(radius * math.cos(angle).toFloat, 0f, radius * math.sin(angle).toFloat)
    }
    Curve(points = pts, radius = tubeRadius, closed = true, material = material)

  /** Open helix along the Y axis. */
  def helix(
    turns: Int,
    radius: Float,
    pitch: Float,
    tubeRadius: Float,
    pointsPerTurn: Int = 32,
    material: Option[Material] = None
  ): Curve =
    val total = turns * pointsPerTurn
    val pts = (0 to total).map { i =>
      val t = 2f * math.Pi.toFloat * i / pointsPerTurn
      Vec3(radius * math.cos(t).toFloat, pitch * i / pointsPerTurn, radius * math.sin(t).toFloat)
    }
    Curve(points = pts, radius = tubeRadius, closed = false, material = material)

/** L-system (Lindenmayer system) fractal object.
  *
  * Generates a 3D branching structure using a grammar string, turtle interpreter,
  * and geometry primitives (curves, spheres, etc.). Supports CLI presets and full
  * DSL parameterization.
  *
  * @param axiom          Starting string for L-system grammar
  * @param rules          Production rules mapping symbols to weighted replacement strings
  * @param iterations     Number of rewriting iterations (0-12)
  * @param angleDegrees   Turtle turn angle in degrees
  * @param segmentLength  Length of each forward segment
  * @param initialWidth   Starting width for curve segments
  * @param widthDecay     Width multiplier per branching step
  * @param seed           Random seed for stochastic rules
  * @param normalizeScale Whether to normalize output to unit cube
  */
case class LSystem(
  axiom: String,
  rules: Map[Char, String],
  iterations: Int = 4,
  angleDegrees: Degrees = Degrees(25.7f),
  segmentLength: Float = 0.1f,
  initialWidth: Float = 0.1f,
  widthDecay: Float = 0.7f,
  seed: Long = 42L,
  normalizeScale: Boolean = true,
  pos: Vec3 = Vec3.Zero,
  size: Float = 1.0f,
  material: Option[Material] = None,
  color: Option[Color] = None,
  ior: Float = 1.0f,
  texture: Option[String] = None,
  videoTexture: Option[menger.video.VideoTexture] = None,
  normalMap: Option[String] = None,
  roughnessMap: Option[String] = None,
  proceduralType: Int = 0,
  proceduralScale: Float = 1.0f,
  rotation: Vec3 = Vec3.Zero
) extends SceneObject:
  FieldConstraints.check("LSystem", "iterations", iterations)
  require(angleDegrees.value > 0, s"angle must be positive, got ${angleDegrees.value}")
  require(segmentLength > 0, "segmentLength must be positive")
  require(initialWidth > 0, "initialWidth must be positive")
  require(rules.nonEmpty, "rules must not be empty")
  require(axiom.nonEmpty, "axiom must not be empty")

  def toObjectSpec: ObjectSpec =
    baseObjectSpec("lsystem",
      level = Some(iterations.toFloat),
      curveData = None
    )

object LSystem:
  def preset(name: String, iterations: Int = 4, size: Float = 1.0f,
    seed: Long = 42L): LSystem =
    val lowerName = name.toLowerCase
    val (axiom, rules, angle, segLen, initWidth, decay, defaultIters) =
      LSystemPresets(lowerName)
    val iters = if iterations > 0 then iterations else defaultIters
    LSystem(axiom, rules, iters, angle, segLen, initWidth, decay,
      seed = seed, size = size)
