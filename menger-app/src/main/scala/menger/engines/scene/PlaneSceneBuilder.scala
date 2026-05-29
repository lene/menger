package menger.engines.scene

import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.common.Vector

class PlaneSceneBuilder(textureDir: String = ".") extends SceneBuilder:

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
    val textureIndices = TextureManager.loadTextures(specs, renderer, textureDir)
    specs.foreach { spec =>
      val material = MaterialExtractor.extract(spec)
      val (nx, ny, nz) = spec.normal.getOrElse((0f, 1f, 0f))
      val d = spec.distance.getOrElse {
        val n = Vector[3](nx, ny, nz)
        val dot = n(0) * spec.x + n(1) * spec.y + n(2) * spec.z
        dot
      }
      val normal = Vector[3](nx, ny, nz)
      val instanceId = renderer.addPlaneInstance(normal, d, material, spec.color2.orNull, spec.checkerSize)
      if instanceId >= 0 then
        applyInstanceTextures(instanceId, spec, textureIndices, renderer)
        logger.debug(s"Added plane instance $instanceId normal=($nx,$ny,$nz) distance=$d")
      else
        logger.error(s"Failed to add plane instance normal=($nx,$ny,$nz) distance=$d")
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    spec1.objectType.toLowerCase == "plane" && spec2.objectType.toLowerCase == "plane"

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long = specs.length.toLong
