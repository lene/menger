package menger.engines.scene

import scala.util.Try

import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.ObjectType
import menger.common.Vector
import menger.optix.OptiXRenderer

class Sierpinski4DSceneBuilder(
  textureDir: String = ".",
  sierpinski4DRecorder: (Int, Int) => Unit = (_, _) => ()
) extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then Left("Object specs list cannot be empty")
    else if !specs.forall(s => ObjectType.isSierpinski4D(s.objectType)) then
      Left("All objects must be sierpinski4d for Sierpinski4DSceneBuilder")
    else if specs.exists(_.level.isEmpty) then
      Left("All sierpinski4d objects require a level parameter")
    else if specs.exists(s => s.level.exists(l => l != l.toInt)) then
      Left("Sierpinski4D does not support fractional levels")
    else Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] = Try:
    logger.debug(s"Setting up ${specs.length} sierpinski4d instances")
    specs.zipWithIndex.foreach { case (spec, specIdx) =>
      val proj     = spec.projection4D.getOrElse(Projection4DSpec.default)
      val material = MaterialExtractor.extract(spec)
      val position = Vector[3](spec.x, spec.y, spec.z)
      val level    = spec.level.get.toInt

      renderer.addSierpinski4DInstance(
        level, position, spec.size,
        proj.eyeW, proj.screenW, proj.rotXW, proj.rotYW, proj.rotZW,
        material
      ) match
        case Some(id) =>
          sierpinski4DRecorder(specIdx, id)
          logger.debug(s"Added sierpinski4d instance $id level=$level")
        case None =>
          logger.error(s"Failed to add sierpinski4d instance at (${spec.x},${spec.y},${spec.z})")
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    ObjectType.isSierpinski4D(spec1.objectType) && ObjectType.isSierpinski4D(spec2.objectType)

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    specs.length.toLong
