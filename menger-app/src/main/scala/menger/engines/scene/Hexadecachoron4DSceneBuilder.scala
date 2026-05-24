package menger.engines.scene

import scala.util.Try

import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.ObjectType
import menger.common.Vector
import menger.optix.OptiXRenderer

class Hexadecachoron4DSceneBuilder(
  textureDir: String = ".",
  hexadecachoron4DRecorder: (Int, Int) => Unit = (_, _) => ()
) extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then Left("Object specs list cannot be empty")
    else if !specs.forall(s => ObjectType.isHexadecachoron4D(s.objectType)) then
      Left("All objects must be hexadecachoron4d for Hexadecachoron4DSceneBuilder")
    else if specs.exists(_.level.isEmpty) then
      Left("All hexadecachoron4d objects require a level parameter")
    else Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] = Try:
    logger.debug(s"Setting up ${specs.length} hexadecachoron4d instances")
    specs.zipWithIndex.foreach { case (spec, specIdx) =>
      val proj     = spec.projection4D.getOrElse(Projection4DSpec.default)
      val material = MaterialExtractor.extract(spec)
      val position = Vector[3](spec.x, spec.y, spec.z)
      val rawLevel = spec.level.get

      def addInstance(level: Int, mat: menger.common.Material, scale: Float): Unit =
        val instanceId = renderer.addHexadecachoron4DInstance(
          level, position, scale,
          proj.eyeW, proj.screenW, proj.rotXW, proj.rotYW, proj.rotZW,
          mat
        )
        if instanceId >= 0 then
          hexadecachoron4DRecorder(specIdx, instanceId)
          logger.debug(s"Added hexadecachoron4d instance $instanceId level=$level")
        else
          logger.error(s"Failed to add hexadecachoron4d instance at (${spec.x},${spec.y},${spec.z})")

      if isFractional(rawLevel) then
        val frac      = rawLevel - rawLevel.floor
        val coarseMat = material.copy(color = material.color.copy(a = material.color.a * (1f - frac)))
        addInstance(rawLevel.floor.toInt + 1, material,  spec.size)
        addInstance(rawLevel.floor.toInt,     coarseMat, spec.size * (1f - CoarseScaleOffset))
      else
        addInstance(rawLevel.toInt, material, spec.size)
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    ObjectType.isHexadecachoron4D(spec1.objectType) && ObjectType.isHexadecachoron4D(spec2.objectType)

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    specs.iterator.map(s => if s.level.exists(isFractional) then 2L else 1L).sum

  private val CoarseScaleOffset = 0.001f
  private def isFractional(level: Float): Boolean = level != level.floor
