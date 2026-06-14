package menger.engines.scene

import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.common.Vector

/**
 * Scene builder for cone primitives.
 *
 * Cones are specified with apex, base center, and radius.
 * If apex/base are not given, they are derived from pos and size.
 */
class ConeSceneBuilder(textureDir: String = ".") extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then Left("Object specs list cannot be empty")
    else if specs.length > maxInstances then
      Left(s"Too many objects: ${specs.length} exceeds max instances limit of $maxInstances. " +
        "Use --max-instances to increase the limit.")
    else if !specs.forall(_.objectType.toLowerCase == "cone") then
      Left("All objects must be cones for ConeSceneBuilder")
    else Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] = Try:
    logger.debug(s"Setting up ${specs.length} cone instances")
    val textureIndices = TextureManager.loadTextures(specs, renderer, textureDir)
    specs.foreach { spec =>
      val material = MaterialExtractor.extract(spec)
      val (ax, ay, az) = spec.apex.getOrElse((spec.x, spec.y + spec.size / 2f, spec.z))
      val (bx, by, bz) = spec.base.getOrElse((spec.x, spec.y - spec.size / 2f, spec.z))
      val r = spec.radius.getOrElse(spec.size / 2f)
      val apex = Vector[3](ax, ay, az)
      val base = Vector[3](bx, by, bz)
      val instanceId = requireInstanceId(
        renderer.addConeInstance(apex, base, r, material),
        s"cone instance at apex=($ax,$ay,$az)"
      )
      applyInstanceTextures(instanceId, spec, textureIndices, renderer)
      logger.debug(s"Added cone instance $instanceId at apex=($ax,$ay,$az) base=($bx,$by,$bz) radius=$r")
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    spec1.objectType.toLowerCase == "cone" && spec2.objectType.toLowerCase == "cone"

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long = specs.length.toLong
