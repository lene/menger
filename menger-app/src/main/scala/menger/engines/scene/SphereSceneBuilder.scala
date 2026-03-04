package menger.engines.scene

import scala.util.Try

import menger.ObjectSpec
import menger.common.TransformUtil
import menger.optix.OptiXRenderer

/**
 * Scene builder for multiple sphere instances using IAS (Instance Acceleration Structure).
 *
 * Creates a scene with multiple sphere instances, each with:
 * - Position and scale (via transform matrix)
 * - Material properties (color, IOR, roughness, metallic, etc.)
 *
 * Key characteristics:
 * - No base geometry setup needed - addSphereInstance() enables IAS mode automatically
 * - All spheres are mutually compatible (no geometry matching required)
 * - Instance count = specs.length (1:1 mapping)
 * - No texture support for spheres
 *
 * Ported from OptiXEngine.setupMultipleSpheres() (lines 280-297).
 */
class SphereSceneBuilder extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then
      Left("Object specs list cannot be empty")
    else if specs.length > maxInstances then
      Left(s"Too many objects: ${specs.length} exceeds max instances limit of $maxInstances. " +
        "Use --max-instances to increase the limit.")
    else if !specs.forall(_.objectType == "sphere") then
      Left("All objects must be spheres for SphereSceneBuilder")
    else
      Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] = Try:
    logger.debug(s"Setting up ${specs.length} sphere instances")

    // addSphereInstance() automatically enables IAS mode - do NOT call setSphere() first!
    specs.foreach { spec =>
      val material = MaterialExtractor.extract(spec)
      val scale = spec.size

      val transform =
        if spec.rotX == 0f && spec.rotY == 0f && spec.rotZ == 0f then
          TransformUtil.createScaleTranslation(scale, spec.x, spec.y, spec.z)
        else
          TransformUtil.createEulerRotationScaleTranslation(spec.rotX, spec.rotY, spec.rotZ, scale, spec.x, spec.y, spec.z)

      val instanceId = renderer.addSphereInstance(transform, material)

      instanceId match
        case Some(id) =>
          logger.debug(s"Added sphere instance $id at position=(${spec.x}, ${spec.y}, ${spec.z}), scale=$scale, material=$material")
        case None =>
          logger.error(s"Failed to add sphere instance at position=(${spec.x}, ${spec.y}, ${spec.z})")
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    // All spheres are compatible with each other
    spec1.objectType == "sphere" && spec2.objectType == "sphere"

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    specs.length.toLong
