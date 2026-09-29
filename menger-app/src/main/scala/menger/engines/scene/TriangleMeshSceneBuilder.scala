package menger.engines.scene

import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.ObjectType
import menger.common.ProfilingConfig
import menger.common.TransformUtil
import menger.common.Vector

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
  mesh4DRecorder: (Int, Int) => Unit = (_, _) => (),
  // (spec index, instance id) of a fractional 4D sponge's hole-cap instance, whose alpha an
  // animation can then update in place (TrackedMesh4D, F52).
  holeCapsRecorder: (Int, Int) => Unit = (_, _) => ()
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
          buildFractionalGpuOps(spec, baseMaterial)
        else
          List(FractionalOp(MeshFactory.createUpload(spec), baseMaterial))

      ops.foreach { op =>
        // Upload mesh and add instance
        op.plan match
          case MeshUploadPlan.Cpu(data) =>
            renderer.addTriangleMesh(data)
          case MeshUploadPlan.Gpu4D(quads4D, vertsPerFace, proj) =>
            val meshIdx = renderer.setProjectedMesh(
              quads4D, vertsPerFace, uvs = null, // scalafix:ok DisableSyntax.null
              eyeW = proj.eyeW, screenW = proj.screenW,
              rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW,
              centerX = 0f, centerY = 0f, centerZ = 0f
            )
            mesh4DRecorder(specIdx, meshIdx)

        val textureIndex = spec.imageTextureKey.flatMap(textureIndices.get).getOrElse(-1)

        val instanceId =
          if ObjectType.isRecursiveIASSponge(spec.objectType) then
            val transform = TransformUtil.createEulerRotationScaleTranslation(
              spec.rotX, spec.rotY, spec.rotZ, spec.size, spec.x, spec.y, spec.z
            )
            val rawLevel = spec.level.get
            if isFractional(rawLevel) then
              val frac      = rawLevel - rawLevel.floor
              val coarseMat = op.material.copy(color = op.material.color.copy(a = op.material.color.a * (1f - frac)))
              val coarseId = requireInstanceId(
                renderer.addRecursiveIASSpongeInstance(
                  rawLevel.floor.toInt, transform, coarseMat, textureIndex
                ),
                s"coarse fractional sponge instance level=${rawLevel.floor.toInt} for ${spec.objectType}"
              )
              applyInstanceTextures(coarseId, spec, textureIndices, renderer)
              requireInstanceId(
                renderer.addRecursiveIASSpongeInstance(
                  rawLevel.floor.toInt + 1, transform, op.material, textureIndex
                ),
                s"fractional sponge instance level=${rawLevel.floor.toInt + 1} for ${spec.objectType}"
              )
            else
              requireInstanceId(
                renderer.addRecursiveIASSpongeInstance(
                  rawLevel.toInt, transform, op.material, textureIndex
                ),
                s"recursive-IAS sponge instance level=${rawLevel.toInt} for ${spec.objectType}"
              )
          else if spec.rotX == 0f && spec.rotY == 0f && spec.rotZ == 0f then
            requireInstanceId(
              renderer.addTriangleMeshInstance(Vector[3](spec.x, spec.y, spec.z), op.material, textureIndex),
              s"${spec.objectType} instance at position=(${spec.x}, ${spec.y}, ${spec.z})"
            )
          else
            val transform = TransformUtil.createEulerRotationScaleTranslation(
              spec.rotX, spec.rotY, spec.rotZ, 1f, spec.x, spec.y, spec.z
            )
            requireInstanceId(
              renderer.addTriangleMeshInstance(transform, op.material, textureIndex),
              s"${spec.objectType} instance at position=(${spec.x}, ${spec.y}, ${spec.z})"
            )
        applyInstanceTextures(instanceId, spec, textureIndices, renderer)
        if op.isHoleCaps then holeCapsRecorder(specIdx, InstanceId.raw(instanceId))
        val levelInfo = spec.level.map(l => f"level=$l%.2f").getOrElse("")
        val textureInfo = if textureIndex >= 0 then s", texture=$textureIndex" else ""
        logger.debug(s"Added ${spec.objectType} instance $instanceId ($levelInfo) at position=(${spec.x}, ${spec.y}, ${spec.z})$textureInfo")
      }
    }

  /** GPU-projected fractional 4D sponge: emit two meshes sharing the projection
    * params: level n+1 fully opaque, and the hole caps of level n (the centre
    * third of each face, which level n+1 leaves open) with alpha = 1 - fractional,
    * so the new holes fade in. Same design as the CPU path's
    * `FractionalLevelSponge`; no face of the caps overlaps level n+1 (usability
    * review 2026-09, session 2, F35). */
  private def buildFractionalGpuOps(
    spec: ObjectSpec,
    baseMaterial: menger.common.Material
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
    List(
      FractionalOp(MeshFactory.createUpload(nextLevelSpec), baseMaterial),
      FractionalOp(
        MeshFactory.createUpload(currentLevelSpec, holeCaps = true),
        TriangleMeshSceneBuilder.holeCapsMaterial(baseMaterial, level),
        isHoleCaps = true
      )
    )

  private final case class FractionalOp(
    plan: MeshUploadPlan,
    material: menger.common.Material,
    isHoleCaps: Boolean = false
  )

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
    // GPU fractional path: 2 instances per fractional spec (level n + level n+1).
    specs.iterator.map { spec =>
      if ObjectType.isRecursiveIASSponge(spec.objectType) && spec.level.exists(isFractional) then 2L
      else if isFractional4DSponge(spec) then 2L
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

object TriangleMeshSceneBuilder:
  /** The hole caps of a fractional 4D sponge fade out as the level rises: alpha = base alpha
    * x (1 - fractional part). Shared by the build and by in-place animation updates
    * (TrackedMesh4D), so both give the same material. */
  def holeCapsMaterial(base: menger.common.Material, level: Float): menger.common.Material =
    base.copy(color = base.color.copy(a = base.color.a * (1f - (level - level.floor))))
