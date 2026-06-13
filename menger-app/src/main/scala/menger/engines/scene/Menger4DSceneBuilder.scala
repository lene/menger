package menger.engines.scene

import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.ObjectType
import menger.common.Vector

class Menger4DSceneBuilder(
  textureDir: String = ".",
  menger4DRecorder: (Int, InstanceId) => Unit = (_, _) => ()
) extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then Left("Object specs list cannot be empty")
    else if !specs.forall(s => ObjectType.isMenger4D(s.objectType)) then
      Left("All objects must be menger4d for Menger4DSceneBuilder")
    else if specs.exists(_.level.isEmpty) then
      Left("All menger4d objects require a level parameter")
    else if specs.exists(levelOutOfRange) then
      Left(s"menger4d level must be in [$MinLevel, $MaxLevel]")
    else Right(())

  /** A fractional level renders two instances (floor and floor+1), so the
    * effective maximum int level handed to the native layer is floor+1. */
  private def levelOutOfRange(spec: ObjectSpec): Boolean =
    spec.level.exists { level =>
      val maxIntLevel = if isFractional(level) then level.floor.toInt + 1 else level.toInt
      level < MinLevel || maxIntLevel > MaxLevel
    }

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] = Try:
    logger.debug(s"Setting up ${specs.length} menger4d instances")
    specs.zipWithIndex.foreach { case (spec, specIdx) =>
      val threshold = spec.distanceThreshold.getOrElse(2)
      val proj      = spec.projection4D.getOrElse(Projection4DSpec.default)
      val material  = MaterialExtractor.extract(spec)
      val position  = Vector[3](spec.x, spec.y, spec.z)
      val rawLevel  = spec.level.get

      def addInstance(level: Int, mat: menger.common.Material, scale: Float): Unit =
        val instanceId = requireInstanceId(
          renderer.addMenger4DInstance(
            level, threshold, position, scale,
            proj.eyeW, proj.screenW, proj.rotXW, proj.rotYW, proj.rotZW,
            mat
          ),
          s"menger4d instance at (${spec.x},${spec.y},${spec.z})"
        )
        menger4DRecorder(specIdx, instanceId)
        logger.debug(s"Added menger4d instance $instanceId level=$level threshold=$threshold")

      if isFractional(rawLevel) then
        val frac      = rawLevel - rawLevel.floor
        val coarseMat = material.copy(color = material.color.copy(a = material.color.a * (1f - frac)))
        addInstance(rawLevel.floor.toInt + 1, material,  spec.size)
        addInstance(rawLevel.floor.toInt,     coarseMat, spec.size * (1f - CoarseScaleOffset))
      else
        addInstance(rawLevel.toInt, material, spec.size)
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    ObjectType.isMenger4D(spec1.objectType) && ObjectType.isMenger4D(spec2.objectType)

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    specs.iterator.map { spec =>
      if spec.level.exists(isFractional) then 2L else 1L
    }.sum

  private val CoarseScaleOffset = 0.001f
  // Recursion-depth bounds; must match MIN_4D_LEVEL / MAX_4D_LEVEL in OptiXWrapper.cpp.
  private val MinLevel = 0
  private val MaxLevel = 14
  private def isFractional(level: Float): Boolean = level != level.floor
