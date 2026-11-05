package menger.engines.scene

import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.common.Material
import menger.common.TransformUtil
import menger.common.Vector
import menger.engines.scene.AxialPrimitiveSceneBuilder.Axis
import menger.engines.scene.AxialPrimitiveSceneBuilder.ExplicitAxis
import menger.engines.scene.AxialPrimitiveSceneBuilder.Point3
import menger.spec.ObjectSpec

object AxialPrimitiveSceneBuilder:
  type Point3 = (Float, Float, Float)

  /** The start and end points of a primitive's axis in world coordinates and its radius, each
    * absent when it is to be derived from the spec's `pos`, `size` and `rotation`. */
  final case class ExplicitAxis(
    start: Option[Point3], end: Option[Point3], radius: Option[Float]
  )

  /** Where a primitive goes: the start and end points of its axis in world coordinates and its
    * radius. */
  final case class Axis(start: Point3, end: Point3, radius: Float)

/**
 * Scene builder for an analytic primitive placed by the two end points of its axis and a radius
 * (cone, cylinder), one instance per spec.
 *
 * Explicit end points are absolute world coordinates and bypass pos/size/rotation. Without them
 * the axis starts `size / 2` below `pos` and ends `size / 2` above it, along the object's rotated
 * y axis, and the radius defaults to `size / 2`.
 */
abstract class AxialPrimitiveSceneBuilder(typeName: String, textureDir: String)
    extends SceneBuilder:

  private[scene] def explicitAxisOf(spec: ObjectSpec): ExplicitAxis

  protected def addInstance(
    renderer: OptiXRenderer, start: Vector[3], end: Vector[3], radius: Float, material: Material
  ): InstanceId

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then Left("Object specs list cannot be empty")
    else if specs.length > maxInstances then
      Left(s"Too many objects: ${specs.length} exceeds max instances limit of $maxInstances. " +
        "Use --max-instances to increase the limit.")
    else if !specs.forall(isOwnType) then
      Left(s"All objects must be ${typeName}s for ${getClass.getSimpleName}")
    else Right(())

  /** Rotates a point that is local to `spec`'s position by the object's Euler rotation
    * (radians), then translates by `spec`'s position — same rotation convention as
    * `TransformUtil.createEulerRotationScaleTranslation`, reused rather than re-derived. */
  private[scene] def rotatedWorldPoint(
    spec: ObjectSpec, localX: Float, localY: Float, localZ: Float
  ): Point3 =
    val m = TransformUtil.createEulerRotationScaleTranslation(
      spec.rotX, spec.rotY, spec.rotZ, 1f, spec.x, spec.y, spec.z
    )
    (
      m(0) * localX + m(1) * localY + m(2) * localZ + m(3),
      m(4) * localX + m(5) * localY + m(6) * localZ + m(7),
      m(8) * localX + m(9) * localY + m(10) * localZ + m(11)
    )

  /** `spec`'s axis and radius in world coordinates: the explicit ones where given, else the
    * defaults derived from `pos`, `size` and `rotation`. (Rotation applies to the defaults only;
    * a cone once dropped it there, Sprint 36 H5.2.) */
  private[scene] def axisOf(spec: ObjectSpec): Axis =
    val explicit = explicitAxisOf(spec)
    Axis(
      explicit.start.getOrElse(rotatedWorldPoint(spec, 0f, -spec.size / 2f, 0f)),
      explicit.end.getOrElse(rotatedWorldPoint(spec, 0f, spec.size / 2f, 0f)),
      explicit.radius.getOrElse(spec.size / 2f)
    )

  override def buildScene(
    specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int
  ): Try[Unit] = Try:
    logger.debug(s"Setting up ${specs.length} $typeName instances")
    val textureIndices = TextureManager.loadTextures(specs, renderer, textureDir)
    specs.foreach { spec =>
      val axis = axisOf(spec)
      val instanceId = addInstance(
        renderer, toVector(axis.start), toVector(axis.end), axis.radius,
        MaterialExtractor.extract(spec)
      )
      applyInstanceTextures(instanceId, spec, textureIndices, renderer)
      logger.debug(s"Added $typeName instance $instanceId at $axis")
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    isOwnType(spec1) && isOwnType(spec2)

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long = specs.length.toLong

  private def isOwnType(spec: ObjectSpec): Boolean = spec.objectType.toLowerCase == typeName

  private def toVector(point: Point3): Vector[3] = Vector[3](point._1, point._2, point._3)
