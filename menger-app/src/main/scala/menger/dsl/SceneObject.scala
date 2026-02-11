package menger.dsl

import menger.ObjectSpec
import menger.common.ObjectType
import scala.annotation.targetName

/** Base trait for all scene objects */
sealed trait SceneObject:
  def pos: Vec3
  def size: Float
  def toObjectSpec: ObjectSpec

/** Sphere object */
case class Sphere(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec =
    ObjectSpec(
      objectType = "sphere",
      x = pos.x,
      y = pos.y,
      z = pos.z,
      size = size,
      level = None,
      color = color.map(_.toCommonColor),
      ior = material.map(_.ior).getOrElse(ior),
      material = material.map(_.toOptixMaterial),
      texture = texture
    )

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

  // Tuple position overloads
  @targetName("sphereFloatTupleMat")
  def apply(pos: (Float, Float, Float), material: Material): Sphere =
    Sphere(Vec3(pos._1, pos._2, pos._3), Some(material))

  @targetName("sphereIntTupleMat")
  def apply(pos: (Int, Int, Int), material: Material): Sphere =
    Sphere(Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat), Some(material))

  @targetName("sphereDoubleTupleMat")
  def apply(pos: (Double, Double, Double), material: Material): Sphere =
    Sphere(Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat), Some(material))

/** Cube object */
case class Cube(
  pos: Vec3 = Vec3.Zero,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None
) extends SceneObject:
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec =
    ObjectSpec(
      objectType = "cube",
      x = pos.x,
      y = pos.y,
      z = pos.z,
      size = size,
      level = None,
      color = color.map(_.toCommonColor),
      ior = material.map(_.ior).getOrElse(ior),
      material = material.map(_.toOptixMaterial),
      texture = texture
    )

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

  // Tuple position overloads
  @targetName("cubeFloatTupleMat")
  def apply(pos: (Float, Float, Float), material: Material): Cube =
    Cube(Vec3(pos._1, pos._2, pos._3), Some(material))

  @targetName("cubeIntTupleMat")
  def apply(pos: (Int, Int, Int), material: Material): Cube =
    Cube(Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat), Some(material))

  @targetName("cubeDoubleTupleMat")
  def apply(pos: (Double, Double, Double), material: Material): Cube =
    Cube(Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat), Some(material))

/** Sponge type enumeration for DSL */
enum SpongeType(val objectTypeName: String):
  case VolumeFilling extends SpongeType("sponge-volume")
  case SurfaceUnfolding extends SpongeType("sponge-surface")
  case CubeSponge extends SpongeType("cube-sponge")

/** Menger sponge fractal object */
case class Sponge(
  spongeType: SpongeType,
  pos: Vec3 = Vec3.Zero,
  level: Float,
  material: Option[Material] = None,
  color: Option[Color] = None,
  size: Float = 1.0f,
  ior: Float = 1.0f,
  texture: Option[String] = None
) extends SceneObject:
  require(level >= 0f, s"Level must be non-negative, got $level")
  require(size > 0f, s"Size must be positive, got $size")
  require(ior >= 0f, s"IOR must be non-negative, got $ior")

  def toObjectSpec: ObjectSpec =
    ObjectSpec(
      objectType = ObjectType.normalize(spongeType.objectTypeName),
      x = pos.x,
      y = pos.y,
      z = pos.z,
      size = size,
      level = Some(level),
      color = color.map(_.toCommonColor),
      ior = material.map(_.ior).getOrElse(ior),
      material = material.map(_.toOptixMaterial),
      texture = texture
    )

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

  // Tuple position overloads
  @targetName("spongeFloatTupleTypeLevel")
  def apply(pos: (Float, Float, Float), spongeType: SpongeType, level: Float): Sponge =
    Sponge(spongeType, Vec3(pos._1, pos._2, pos._3), level)

  @targetName("spongeIntTupleTypeLevel")
  def apply(pos: (Int, Int, Int), spongeType: SpongeType, level: Float): Sponge =
    Sponge(spongeType, Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat), level)

  @targetName("spongeDoubleTupleTypeLevel")
  def apply(pos: (Double, Double, Double), spongeType: SpongeType, level: Float): Sponge =
    Sponge(spongeType, Vec3(pos._1.toFloat, pos._2.toFloat, pos._3.toFloat), level)

// Export SpongeType values for convenient imports
export SpongeType.{VolumeFilling, SurfaceUnfolding, CubeSponge}
