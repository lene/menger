package menger.engines.scene

import scala.util.Try

import menger.ObjectSpec
import menger.ProfilingConfig
import menger.common.ObjectType
import menger.common.Vector
import menger.optix.OptiXRenderer

/**
 * Scene builder for multiple triangle mesh instances with optional textures.
 *
 * Creates a scene with multiple mesh instances sharing the same base geometry:
 * - Supported mesh types: cube, sponge-volume, sponge-surface, tesseract
 * - Each instance has position, material, and optional texture
 * - All instances must use compatible geometry (same type + parameters)
 *
 * Key characteristics:
 * - Base mesh set once (shared by all instances)
 * - Strict compatibility validation - all specs must match geometry type/params
 * - Instance count = specs.length (1:1 mapping)
 * - Optional texture support per instance
 * - Sponges must have same level
 * - Hypercubes must have same 4D projection parameters
 *
 * Ported from OptiXEngine.setupMultipleTriangleMeshes() (lines 299-335)
 * and isCompatibleMesh() (lines 345-363).
 */
class TriangleMeshSceneBuilder(textureDir: String)(using profilingConfig: ProfilingConfig)
  extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then
      Left("Object specs list cannot be empty")
    else if specs.length > maxInstances then
      Left(s"Too many objects: ${specs.length} exceeds max instances limit of $maxInstances. " +
        "Use --max-instances to increase the limit.")
    else if !specs.forall(isTriangleMeshType) then
      Left("All objects must be triangle mesh types (cube, sponge-*, tesseract)")
    else
      // Check compatibility - all specs must be compatible with first spec
      val firstSpec = specs.head
      specs.find(!isCompatible(_, firstSpec)) match
        case Some(incompatible) =>
          Left(s"Incompatible mesh types or parameters: ${firstSpec.objectType} vs ${incompatible.objectType}. " +
            "All triangle mesh objects must be the same type with matching parameters.")
        case None =>
          Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer): Try[Unit] = Try:
    logger.debug(s"Setting up ${specs.length} triangle mesh instances")

    // Create shared base geometry
    val firstSpec = specs.head
    val mesh = MeshFactory.create(firstSpec)
    renderer.setTriangleMesh(mesh)

    // Load textures
    val textureIndices = TextureManager.loadTextures(specs, renderer, textureDir)

    // Add instances
    specs.foreach { spec =>
      val material = MaterialExtractor.extract(spec)
      val position = Vector[3](spec.x, spec.y, spec.z)

      // Get texture index if this spec has a texture
      val textureIndex = spec.texture.flatMap(textureIndices.get).getOrElse(-1)

      val instanceId = renderer.addTriangleMeshInstance(position, material, textureIndex)

      instanceId match
        case Some(id) =>
          val textureInfo = if textureIndex >= 0 then s", texture=$textureIndex" else ""
          logger.debug(s"Added ${spec.objectType} instance $id at position=(${spec.x}, ${spec.y}, ${spec.z}), material=$material$textureInfo")
        case None =>
          logger.error(s"Failed to add ${spec.objectType} instance at position=(${spec.x}, ${spec.y}, ${spec.z})")
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    (spec1.objectType, spec2.objectType) match
      case (t1, t2) if t1 == t2 =>
        // Same type - check if parameters match
        if ObjectType.isSponge(t1) then
          // Sponges must have same level
          (spec1.level, spec2.level) match
            case (Some(l1), Some(l2)) => l1 == l2
            case _ => false  // Missing level
        else if ObjectType.isHypercube(t1) then
          // Hypercubes must have same 4D projection params
          (spec1.projection4D, spec2.projection4D) match
            case (Some(p1), Some(p2)) =>
              p1.eyeW == p2.eyeW && p1.screenW == p2.screenW &&
              p1.rotXW == p2.rotXW && p1.rotYW == p2.rotYW && p1.rotZW == p2.rotZW
            case (None, None) => true  // Both using defaults
            case _ => false
        else
          true  // Non-sponge, non-hypercube types are always compatible with same type
      case _ => false  // Different types

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    specs.length.toLong

  private def isTriangleMeshType(spec: ObjectSpec): Boolean =
    spec.objectType == "cube" || ObjectType.isSponge(spec.objectType) || ObjectType.isHypercube(spec.objectType)
