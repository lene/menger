package menger.engines.scene

import scala.util.Try

import menger.ObjectSpec
import menger.ProfilingConfig
import menger.Projection4DSpec
import menger.common.ObjectType
import menger.common.TransformUtil
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
 * - Each spec gets its own base mesh (via setTriangleMesh per instance)
 * - Compatibility validation ensures all specs are triangle mesh types
 * - Instance count = specs.length (1:1 mapping)
 * - Optional texture support per instance
 * - 3D sponges: different levels and types (volume/surface) may be mixed
 * - Hypercubes must have same 4D projection parameters
 *
 * Ported from OptiXEngine.setupMultipleTriangleMeshes() (lines 299-335)
 * and isCompatibleMesh() (lines 345-363).
 */
class TriangleMeshSceneBuilder(
  textureDir: String,
  gpuProject4D: Boolean = false,
  mesh4DRecorder: (Int, Int) => Unit = (_, _) => ()
)(using profilingConfig: ProfilingConfig)
  extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then
      Left("Object specs list cannot be empty")
    else if !specs.forall(isTriangleMeshType) then
      Left("All objects must be triangle mesh types (cube, sponge-*, tesseract, tetrahedron, octahedron, icosahedron, dodecahedron, parametric)")
    else if specs.exists(invalidRecursiveIASLevel) then
      Left("sponge-recursive-ias requires level in [1, 14)")
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

    // For each spec, emit one or more (upload-plan, material-override?) pairs.
    // Fractional 4D sponges in GPU mode produce two GPU meshes (level n+1 opaque,
    // level n with alpha=1-frac). All other specs produce one entry.
    specs.zipWithIndex.foreach { case (spec, specIdx) =>
      val baseMaterial = MaterialExtractor.extract(spec)
      val ops: List[FractionalOp] =
        if isFractional4DSponge(spec) then
          if gpuProject4D then
            buildFractionalGpuOps(spec, baseMaterial)
          else
            // Legacy CPU-merged path: single mesh with per-vertex alpha.
            logger.debug(s"Creating fractional mesh for ${spec.objectType} level=${spec.level.get}")
            List(FractionalOp(MeshUploadPlan.Cpu(createFractionalMesh(spec)), baseMaterial))
        else
          List(FractionalOp(MeshFactory.createUpload(spec, gpuProject4D), baseMaterial))

      ops.foreach { op =>
        // Upload mesh and add instance
        op.plan match
          case MeshUploadPlan.Cpu(data) =>
            renderer.setTriangleMesh(data)
          case MeshUploadPlan.Gpu4D(quads4D, vertsPerFace, proj) =>
            val meshIdx = renderer.setProjectedMesh(
              quads4D, vertsPerFace, uvs = None,
              eyeW = proj.eyeW, screenW = proj.screenW,
              rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW,
              centerX = 0f, centerY = 0f, centerZ = 0f
            )
            mesh4DRecorder(specIdx, meshIdx)

        val textureIndex = spec.texture.flatMap(textureIndices.get).getOrElse(-1)

        val instanceId =
          if ObjectType.isRecursiveIASSponge(spec.objectType) then
            val transform = TransformUtil.createEulerRotationScaleTranslation(
              spec.rotX, spec.rotY, spec.rotZ, spec.size, spec.x, spec.y, spec.z
            )
            val rawLevel = spec.level.get
            if isFractional(rawLevel) then
              val frac      = rawLevel - rawLevel.floor
              val coarseMat = op.material.copy(color = op.material.color.copy(a = op.material.color.a * (1f - frac)))
              val coarseId  = renderer.addRecursiveIASSpongeInstance(rawLevel.floor.toInt, transform, coarseMat, textureIndex)
              coarseId.foreach(id => applyInstanceTextures(id, spec, textureIndices, renderer))
              renderer.addRecursiveIASSpongeInstance(rawLevel.floor.toInt + 1, transform, op.material, textureIndex)
            else
              renderer.addRecursiveIASSpongeInstance(rawLevel.toInt, transform, op.material, textureIndex)
          else if spec.rotX == 0f && spec.rotY == 0f && spec.rotZ == 0f then
            renderer.addTriangleMeshInstance(Vector[3](spec.x, spec.y, spec.z), op.material, textureIndex)
          else
            val transform = TransformUtil.createEulerRotationScaleTranslation(
              spec.rotX, spec.rotY, spec.rotZ, 1f, spec.x, spec.y, spec.z
            )
            renderer.addTriangleMeshInstance(transform, op.material, textureIndex)
        instanceId match
          case Some(id) =>
            applyInstanceTextures(id, spec, textureIndices, renderer)
            val levelInfo = spec.level.map(l => f"level=$l%.2f").getOrElse("")
            val textureInfo = if textureIndex >= 0 then s", texture=$textureIndex" else ""
            logger.debug(s"Added ${spec.objectType} instance $id ($levelInfo) at position=(${spec.x}, ${spec.y}, ${spec.z})$textureInfo")
          case None =>
            logger.error(s"Failed to add ${spec.objectType} instance at position=(${spec.x}, ${spec.y}, ${spec.z})")
      }
    }

  /** GPU-projected fractional 4D sponge: emit two integer-level meshes
    * (level n+1 fully opaque, level n with alpha = 1 - fractional). Both
    * share projection params; per-mesh material alpha differs.
    *
    * The lower-level (currentLevel) quads are expanded along their 4D face
    * normals by `SkinNormalOffset` before upload — equivalent to the CPU
    * path's `expandAlongNormals` — so that the level-n skin faces do not
    * perfectly overlap the level-(n+1) surface, preventing z-fighting. */
  private def buildFractionalGpuOps(
    spec: ObjectSpec,
    baseMaterial: menger.optix.Material
  )(using profilingConfig: ProfilingConfig): List[FractionalOp] =
    val level = spec.level.get
    val fractionalPart = level - level.floor
    val alphaTransparent = 1.0f - fractionalPart
    val nextLevelSpec = spec.copy(level = Some((level + 1).floor))
    val currentLevelSpec = spec.copy(level = Some(level.floor))
    logger.debug(
      s"GPU fractional split: ${spec.objectType} level=$level → " +
      s"slot[opaque level ${(level + 1).floor}] + slot[level ${level.floor} alpha=$alphaTransparent]"
    )
    val opaqueMaterial = baseMaterial
    val transparentMaterial = baseMaterial.copy(
      color = baseMaterial.color.copy(a = baseMaterial.color.a * alphaTransparent)
    )
    List(
      FractionalOp(MeshFactory.createUpload(nextLevelSpec, gpuProject4D = true), opaqueMaterial),
      FractionalOp(
        MeshFactory.createUpload(currentLevelSpec, gpuProject4D = true,
          skinOffset = FractionalLevelSponge.SkinNormalOffset),
        transparentMaterial
      )
    )

  private final case class FractionalOp(plan: MeshUploadPlan, material: menger.optix.Material)

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
    // TD-5 resolution (Sprint 18.1): each spec gets its own mesh + GAS via per-spec
    // setTriangleMesh + addTriangleMeshInstance, so distinct triangle-mesh types coexist
    // naturally in the IAS. The only remaining cross-spec constraint is that 4D-projected
    // specs must share projection parameters, since projection is a global render setting.
    val t1 = spec1.objectType.toLowerCase
    val t2 = spec2.objectType.toLowerCase

    val spongeLevelsOk =
      (!ObjectType.isSponge(t1) || spec1.level.isDefined) &&
      (!ObjectType.isSponge(t2) || spec2.level.isDefined) &&
      (!ObjectType.is4DSponge(t1) || spec1.level.isDefined) &&
      (!ObjectType.is4DSponge(t2) || spec2.level.isDefined)

    val projectionOk =
      if ObjectType.isProjected4D(t1) && ObjectType.isProjected4D(t2) then
        matchingProjectionParams(spec1, spec2)
      else true

    spongeLevelsOk && projectionOk

  private def matchingProjectionParams(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    val p1 = spec1.projection4D.getOrElse(Projection4DSpec.default)
    val p2 = spec2.projection4D.getOrElse(Projection4DSpec.default)
    p1 == p2

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    // CPU-merged fractional path: 1 instance per spec.
    // GPU fractional path: 2 instances per fractional spec (level n + level n+1).
    specs.iterator.map { spec =>
      if ObjectType.isRecursiveIASSponge(spec.objectType) && spec.level.exists(isFractional) then 2L
      else if isFractional4DSponge(spec) && gpuProject4D then 2L
      else 1L
    }.sum

  private def isTriangleMeshType(spec: ObjectSpec): Boolean =
    ObjectType.isTriangleMesh(spec.objectType)

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

  private def invalidRecursiveIASLevel(spec: ObjectSpec): Boolean =
    if !ObjectType.isRecursiveIASSponge(spec.objectType) then false
    else spec.level match
      case Some(l) => l < 1f || l >= 14f
      case None => true
