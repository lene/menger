package menger.engines.scene

import scala.util.Try

import io.github.lene.optix.MengerRenderer
import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.Material
import menger.common.ObjectType
import menger.common.Vector

/** Per-IFS-type variation for the single projected-4D scene builder (F9, Sprint 35 Ph4).
  *
  * The three IFS 4D types were three near-identical SceneBuilder classes differing only
  * in the type name, the MengerRenderer add call (menger4d passes a distance threshold;
  * sierpinski4d/hexadecachoron4d derive a hit bias from material alpha inside
  * MengerRenderer), and menger4d's recursion-depth bounds. This descriptor carries just
  * those variations; everything else lives once in [[Instanced4DSceneBuilder]].
  *
  * @param name        canonical (normalized) object-type name, e.g. "menger4d"
  * @param addInstance (renderer, level, distanceThreshold, position, scale, projection, material)
  *                    => raw instance id; distanceThreshold is read only by menger4d
  * @param maxLevel    recursion-depth upper bound (menger4d only); must match
  *                    MAX_4D_LEVEL in OptiXWrapper.cpp. Lower bound is always 0.
  */
case class IFS4DType(
  name: String,
  addInstance: (MengerRenderer, Int, Int, Vector[3], Float, Projection4DSpec, Material) => Int,
  maxLevel: Option[Int] = None
)

object IFS4DType:
  val Menger4D: IFS4DType = IFS4DType("menger4d",
    (r, level, threshold, pos, scale, proj, mat) =>
      r.addMenger4DInstance(level, threshold, pos, scale,
        proj.eyeW, proj.screenW, proj.rotXW, proj.rotYW, proj.rotZW, mat),
    maxLevel = Some(14))

  val Sierpinski4D: IFS4DType = IFS4DType("sierpinski4d",
    (r, level, _, pos, scale, proj, mat) =>
      r.addSierpinski4DInstance(level, pos, scale,
        proj.eyeW, proj.screenW, proj.rotXW, proj.rotYW, proj.rotZW, mat))

  val Hexadecachoron4D: IFS4DType = IFS4DType("hexadecachoron4d",
    (r, level, _, pos, scale, proj, mat) =>
      r.addHexadecachoron4DInstance(level, pos, scale,
        proj.eyeW, proj.screenW, proj.rotXW, proj.rotYW, proj.rotZW, mat))

  val all: List[IFS4DType] = List(Menger4D, Sierpinski4D, Hexadecachoron4D)

/** Builds a homogeneous group of GPU-projected 4D IFS instances (menger4d /
  * sierpinski4d / hexadecachoron4d). One generic path parameterized by [[IFS4DType]];
  * replaces the Menger4D/Sierpinski4D/Hexadecachoron4D builder triplication (F9).
  *
  * A fractional level renders two instances (floor and floor+1, the coarser one slightly
  * shrunk and alpha-faded) for all three types.
  */
class Instanced4DSceneBuilder(
  val ifsType: IFS4DType,
  textureDir: String = ".",
  recorder: (Int, InstanceId) => Unit = (_, _) => ()
) extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then Left("Object specs list cannot be empty")
    else if !specs.forall(s => ObjectType.normalize(s.objectType) == ifsType.name) then
      Left(s"All objects must be ${ifsType.name} for Instanced4DSceneBuilder")
    else if specs.exists(_.level.isEmpty) then
      Left(s"All ${ifsType.name} objects require a level parameter")
    else if ifsType.maxLevel.exists(max => specs.exists(levelOutOfRange(_, max))) then
      Left(s"${ifsType.name} level must be in [0, ${ifsType.maxLevel.get}]")
    else Right(())

  /** A fractional level renders two instances (floor and floor+1), so the
    * effective maximum int level handed to the native layer is floor+1. */
  private def levelOutOfRange(spec: ObjectSpec, maxLevel: Int): Boolean =
    spec.level.exists { level =>
      val maxIntLevel = if isFractional(level) then level.floor.toInt + 1 else level.toInt
      level < 0 || maxIntLevel > maxLevel
    }

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] = Try:
    logger.debug(s"Setting up ${specs.length} ${ifsType.name} instances")
    val mengerRenderer = MengerRenderer.of(renderer)
    specs.zipWithIndex.foreach { case (spec, specIdx) =>
      val threshold = spec.distanceThreshold.getOrElse(2)
      val proj      = spec.projection4D.getOrElse(Projection4DSpec.default)
      val material  = MaterialExtractor.extract(spec)
      val position  = Vector[3](spec.x, spec.y, spec.z)
      val rawLevel  = spec.level.get

      def addInstance(level: Int, mat: Material, scale: Float): Unit =
        val instanceId = requireInstanceId(
          ifsType.addInstance(mengerRenderer, level, threshold, position, scale, proj, mat),
          s"${ifsType.name} instance at (${spec.x},${spec.y},${spec.z})"
        )
        recorder(specIdx, instanceId)
        logger.debug(s"Added ${ifsType.name} instance $instanceId level=$level")

      if isFractional(rawLevel) then
        val frac      = rawLevel - rawLevel.floor
        val coarseMat = material.copy(color = material.color.copy(a = material.color.a * (1f - frac)))
        addInstance(rawLevel.floor.toInt + 1, material,  spec.size)
        addInstance(rawLevel.floor.toInt,     coarseMat, spec.size * (1f - CoarseScaleOffset))
      else
        addInstance(rawLevel.toInt, material, spec.size)
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    ObjectType.normalize(spec1.objectType) == ifsType.name &&
      ObjectType.normalize(spec2.objectType) == ifsType.name

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    specs.iterator.map { spec =>
      if spec.level.exists(isFractional) then 2L else 1L
    }.sum

  private val CoarseScaleOffset = 0.001f
  private def isFractional(level: Float): Boolean = level != level.floor
