package menger.engines.scene

import scala.util.Try

import menger.ObjectSpec
import menger.common.Vector
import menger.optix.OptiXRenderer

class PlaneSceneBuilder extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then Left("Object specs list cannot be empty")
    else if specs.length > maxInstances then
      Left(s"Too many objects: ${specs.length} exceeds max instances limit of $maxInstances. " +
        "Use --max-instances to increase the limit.")
    else if !specs.forall(_.objectType.toLowerCase == "plane") then
      Left("All objects must be planes for PlaneSceneBuilder")
    else Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] = Try:
    logger.debug(s"Setting up ${specs.length} plane instances")
    specs.foreach { spec =>
      val material = MaterialExtractor.extract(spec)
      val (nx, ny, nz) = spec.normal.getOrElse((0f, 1f, 0f))
      val d = spec.distance.getOrElse {
        val n = Vector[3](nx, ny, nz)
        val dot = n(0) * spec.x + n(1) * spec.y + n(2) * spec.z
        dot
      }
      val normal = Vector[3](nx, ny, nz)
      renderer.addPlaneInstance(normal, d, material, spec.color2, spec.checkerSize) match
        case Some(id) =>
          logger.debug(s"Added plane instance $id normal=($nx,$ny,$nz) distance=$d")
        case None =>
          logger.error(s"Failed to add plane instance normal=($nx,$ny,$nz) distance=$d")
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    spec1.objectType.toLowerCase == "plane" && spec2.objectType.toLowerCase == "plane"

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long = specs.length.toLong
