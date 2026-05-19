package menger.engines.scene

import scala.util.Try

import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.ObjectType
import menger.common.Vector
import menger.optix.OptiXRenderer

class Menger4DSceneBuilder(textureDir: String = ".") extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then Left("Object specs list cannot be empty")
    else if !specs.forall(s => ObjectType.isMenger4D(s.objectType)) then
      Left("All objects must be menger4d for Menger4DSceneBuilder")
    else if specs.exists(_.level.isEmpty) then
      Left("All menger4d objects require a level parameter")
    else Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] = Try:
    logger.debug(s"Setting up ${specs.length} menger4d instances")
    specs.foreach { spec =>
      val level     = spec.level.get.toInt
      val threshold = spec.distanceThreshold.getOrElse(2)
      val proj      = spec.projection4D.getOrElse(Projection4DSpec.default)
      val material  = MaterialExtractor.extract(spec)
      val position  = Vector[3](spec.x, spec.y, spec.z)
      renderer.addMenger4DInstance(
        level, threshold, position, spec.size,
        proj.eyeW, proj.screenW, proj.rotXW, proj.rotYW, proj.rotZW,
        material
      ) match
        case Some(id) =>
          logger.debug(s"Added menger4d instance $id level=$level threshold=$threshold")
        case None =>
          logger.error(s"Failed to add menger4d instance at (${spec.x},${spec.y},${spec.z})")
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    ObjectType.isMenger4D(spec1.objectType) && ObjectType.isMenger4D(spec2.objectType)

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long = specs.length.toLong
