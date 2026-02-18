package menger.engines.scene

import scala.util.Try

import menger.ObjectSpec
import menger.ProfilingConfig
import menger.Projection4DSpec
import menger.common.ObjectType
import menger.common.TriangleMeshData
import menger.common.Vector
import menger.objects.FractionalLevelSponge
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
    else if !specs.forall(isTriangleMeshType) then
      Left("All objects must be triangle mesh types (cube, sponge-*, tesseract)")
    else
      // Check instance count (accounting for fractional levels creating 2 instances)
      val instanceCount = calculateInstanceCount(specs)
      if instanceCount > maxInstances then
        Left(s"Too many instances: $instanceCount exceeds max instances limit of $maxInstances. " +
          s"(${specs.length} specs, some with fractional levels). " +
          "Use --max-instances to increase the limit.")
      else
        // Check compatibility - all specs must be compatible with first spec
        val firstSpec = specs.head
        specs.find(!isCompatible(_, firstSpec)) match
          case Some(incompatible) =>
            Left(s"Incompatible mesh types or parameters: ${firstSpec.objectType} vs ${incompatible.objectType}. " +
              "All triangle mesh objects must be the same type with matching parameters.")
          case None =>
            Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] = Try:
    logger.debug(s"Setting up triangle mesh instances for ${specs.length} specs")

    // Load textures once
    val textureIndices = TextureManager.loadTextures(specs, renderer, textureDir)

    // For each spec, create either a single mesh or a merged fractional mesh
    specs.foreach { spec =>
      val mesh = if isFractional4DSponge(spec) then
        // Fractional level: merge both levels with per-vertex alpha
        logger.debug(s"Creating fractional mesh for ${spec.objectType} level=${spec.level.get}")
        createFractionalMesh(spec)
      else
        // Integer level or non-sponge: single mesh
        MeshFactory.create(spec)

      // Upload mesh and add instance
      renderer.setTriangleMesh(mesh)

      val position = Vector[3](spec.x, spec.y, spec.z)
      val textureIndex = spec.texture.flatMap(textureIndices.get).getOrElse(-1)
      val material = MaterialExtractor.extract(spec)

      val instanceId = renderer.addTriangleMeshInstance(position, material, textureIndex)
      instanceId match
        case Some(id) =>
          val levelInfo = spec.level.map(l => f"level=$l%.2f").getOrElse("")
          val textureInfo = if textureIndex >= 0 then s", texture=$textureIndex" else ""
          logger.debug(s"Added ${spec.objectType} instance $id ($levelInfo) at position=(${spec.x}, ${spec.y}, ${spec.z})$textureInfo")
        case None =>
          logger.error(s"Failed to add ${spec.objectType} instance at position=(${spec.x}, ${spec.y}, ${spec.z})")
    }

  /**
   * Create a merged mesh for fractional level rendering.
   *
   * For level=1.5:
   * - Generates level 2 geometry with vertex alpha = 1.0 (opaque)
   * - Generates level 1 geometry with vertex alpha = 0.5 (1.0 - frac)
   * - Merges both into a single mesh
   *
   * The shader will interpolate per-vertex alpha and multiply with material alpha:
   * - Level 2 triangles: final_alpha = 1.0 × material.alpha
   * - Level 1 triangles: final_alpha = 0.5 × material.alpha
   */
  private def createFractionalMesh(spec: ObjectSpec): TriangleMeshData =
    val level = spec.level.get
    val fractionalPart = level - level.floor
    val alphaTransparent = 1.0f - fractionalPart

    // Generate both level geometries
    val nextLevelSpec = spec.copy(level = Some((level + 1).floor))
    val currentLevelSpec = spec.copy(level = Some(level.floor))

    val nextLevel = MeshFactory.create(nextLevelSpec)
    // Expand skin faces outward along normals to prevent z-fighting with the underlying sponge
    // faces (same fix as SpongeByVolume.getFractionalMesh — see there for detailed rationale).
    val currentLevel = TriangleMeshData.expandAlongNormals(
      MeshFactory.create(currentLevelSpec),
      FractionalLevelSponge.SkinNormalOffset
    )

    // Assign per-vertex alpha
    val nextWithAlpha = TriangleMeshData.withAlpha(nextLevel, 1.0f)
    val currentWithAlpha = TriangleMeshData.withAlpha(currentLevel, alphaTransparent)

    // Merge into single mesh
    logger.debug(s"Merging fractional mesh: level ${level.floor} (alpha=$alphaTransparent) + level ${(level+1).floor} (alpha=1.0)")
    TriangleMeshData.merge(Seq(nextWithAlpha, currentWithAlpha))

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    val t1 = spec1.objectType.toLowerCase
    val t2 = spec2.objectType.toLowerCase

    (t1, t2) match
      case (t1, t2) if t1 == t2 =>
        // Same type - check if parameters match
        if ObjectType.isSponge(t1) then
          // 3D sponges: Must have same level (shared geometry)
          (spec1.level, spec2.level) match
            case (Some(l1), Some(l2)) => l1 == l2
            case _ => false  // Missing level
        else if ObjectType.is4DSponge(t1) then
          // 4D sponges: Compatible if both have levels (any levels ok)
          // Different levels and fractional levels handled via multiple geometries
          (spec1.level, spec2.level) match
            case (Some(_), Some(_)) => matchingProjectionParams(spec1, spec2)
            case _ => false  // Missing level
        else if ObjectType.isProjected4D(t1) then
          // Other 4D objects (tesseract): must have matching projection parameters
          matchingProjectionParams(spec1, spec2)
        else
          true  // Other types (cube) always compatible with same type

      // Allow mixing different 4D projected types if projection params match
      case (t1, t2) if ObjectType.isProjected4D(t1) && ObjectType.isProjected4D(t2) =>
        // Both must have levels if they're sponges
        val levelsOk = (ObjectType.is4DSponge(t1), ObjectType.is4DSponge(t2)) match
          case (true, true) => spec1.level.isDefined && spec2.level.isDefined
          case (true, false) => spec1.level.isDefined
          case (false, true) => spec2.level.isDefined
          case (false, false) => true
        levelsOk && matchingProjectionParams(spec1, spec2)

      case _ => false  // Different types

  private def matchingProjectionParams(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    val p1 = spec1.projection4D.getOrElse(Projection4DSpec.default)
    val p2 = spec2.projection4D.getOrElse(Projection4DSpec.default)
    p1 == p2

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    // Per-vertex alpha implementation: all specs create 1 instance (fractional levels are merged)
    specs.length.toLong

  private def isTriangleMeshType(spec: ObjectSpec): Boolean =
    spec.objectType == "cube" || ObjectType.isSponge(spec.objectType) || ObjectType.isProjected4D(spec.objectType)

  /**
   * Check if a spec is a 4D sponge with fractional level.
   */
  private def isFractional4DSponge(spec: ObjectSpec): Boolean =
    ObjectType.is4DSponge(spec.objectType) && spec.level.exists(isFractional)

  /**
   * Check if a level value has a fractional component.
   */
  private def isFractional(level: Float): Boolean =
    level != level.floor
