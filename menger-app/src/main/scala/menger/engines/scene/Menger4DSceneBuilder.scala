package menger.engines.scene

import scala.util.Try

import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.ObjectType
import menger.common.Vector
import menger.optix.OptiXRenderer

class Menger4DSceneBuilder(
  textureDir: String = ".",
  menger4DRecorder: (Int, Int) => Unit = (_, _) => ()
) extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then Left("Object specs list cannot be empty")
    else if !specs.forall(s => ObjectType.isMenger4D(s.objectType)) then
      Left("All objects must be menger4d for Menger4DSceneBuilder")
    else if specs.exists(_.level.isEmpty) then
      Left("All menger4d objects require a level parameter")
    else Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] = Try:
    logger.debug(s"Setting up ${specs.length} menger4d instances")
    specs.zipWithIndex.foreach { case (spec, specIdx) =>
      val threshold = spec.distanceThreshold.getOrElse(2)
      val proj      = spec.projection4D.getOrElse(Projection4DSpec.default)
      val material  = MaterialExtractor.extract(spec)
      val position  = Vector[3](spec.x, spec.y, spec.z)
      val rawLevel  = spec.level.get

      def addInstance(level: Int, mat: menger.optix.Material): Unit =
        renderer.addMenger4DInstance(
          level, threshold, position, spec.size,
          proj.eyeW, proj.screenW, proj.rotXW, proj.rotYW, proj.rotZW,
          mat
        ) match
          case Some(id) =>
            menger4DRecorder(specIdx, id)
            logger.debug(s"Added menger4d instance $id level=$level threshold=$threshold")
          case None =>
            logger.error(s"Failed to add menger4d instance at (${spec.x},${spec.y},${spec.z})")

      if isFractional(rawLevel) then
        val frac          = rawLevel - rawLevel.floor
        val coarseAlpha   = 1.0f - frac
        val coarseMat     = material.copy(color = material.color.copy(a = material.color.a * coarseAlpha))
        addInstance(rawLevel.floor.toInt + 1, material)   // fine level — fully opaque
        addInstance(rawLevel.floor.toInt,     coarseMat)  // coarse skin — fading out
      else
        addInstance(rawLevel.toInt, material)
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    ObjectType.isMenger4D(spec1.objectType) && ObjectType.isMenger4D(spec2.objectType)

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    specs.iterator.map { spec =>
      if spec.level.exists(isFractional) then 2L else 1L
    }.sum

  private def isFractional(level: Float): Boolean = level != level.floor
