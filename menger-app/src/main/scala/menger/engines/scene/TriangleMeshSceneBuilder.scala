package menger.engines.scene

import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
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
  holeCapsRecorder: (Int, Int) => Unit = (_, _) => (),
  // (spec index, instance id) of every triangle-mesh instance, so an animation can move or
  // rotate it in place via setInstanceTransform (TrackedMesh4D, F52).
  instanceRecorder: (Int, Int) => Unit = (_, _) => ()
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
            // The plain cube was uploaded above. addRecursiveIASSpongeInstance wraps the most
            // recently uploaded mesh, so any other leaf (the hole caps) goes up right before
            // its own instance.
            val cube = MeshFactory.create(spec)
            val ids = TriangleMeshSceneBuilder.recursiveIASInstances(spec.level.get, cube, op.material)
              .map { (leaf, level, material, coverage) =>
                if leaf ne cube then
                  val _ = renderer.addTriangleMesh(leaf)
                val id = requireInstanceId(
                  renderer.addRecursiveIASSpongeInstance(level, transform, material, textureIndex),
                  s"recursive-IAS sponge instance level=$level for ${spec.objectType}"
                )
                setCoverage(renderer, id, coverage)
                id
              }
            ids.tail.foreach(applyInstanceTextures(_, spec, textureIndices, renderer))
            ids.head
          else if spec.rotX == 0f && spec.rotY == 0f && spec.rotZ == 0f then
            requireInstanceId(
              renderer.addTriangleMeshInstance(Vector[3](spec.x, spec.y, spec.z), op.material, textureIndex),
              s"${spec.objectType} instance at position=(${spec.x}, ${spec.y}, ${spec.z})"
            )
          else
            requireInstanceId(
              renderer.addTriangleMeshInstance(
                TriangleMeshSceneBuilder.instanceTransform(spec), op.material, textureIndex
              ),
              s"${spec.objectType} instance at position=(${spec.x}, ${spec.y}, ${spec.z})"
            )
        setCoverage(renderer, instanceId, op.coverage)
        applyInstanceTextures(instanceId, spec, textureIndices, renderer)
        instanceRecorder(specIdx, InstanceId.raw(instanceId))
        if op.isHoleCaps then holeCapsRecorder(specIdx, InstanceId.raw(instanceId))
        val levelInfo = spec.level.map(l => f"level=$l%.2f").getOrElse("")
        val textureInfo = if textureIndex >= 0 then s", texture=$textureIndex" else ""
        logger.debug(s"Added ${spec.objectType} instance $instanceId ($levelInfo) at position=(${spec.x}, ${spec.y}, ${spec.z})$textureInfo")
      }
    }

  /** GPU-projected fractional 4D sponge: emit two meshes sharing the projection
    * params: level n+1 fully present, and the hole caps of level n (the centre
    * third of each face, which level n+1 leaves open) at coverage 1 - fractional,
    * so the new holes fade in. Same design as the CPU path's
    * `FractionalLevelSponge`; no face of the caps overlaps level n+1 (usability
    * review 2026-09, session 2, F35). */
  private def buildFractionalGpuOps(
    spec: ObjectSpec,
    baseMaterial: menger.common.Material
  )(using profilingConfig: ProfilingConfig): List[FractionalOp] =
    val level = spec.level.get
    val coverage = TriangleMeshSceneBuilder.holeCapsCoverage(level)
    val nextLevelSpec = spec.copy(level = Some((level + 1).floor))
    val currentLevelSpec = spec.copy(level = Some(level.floor))
    logger.debug(
      s"GPU fractional split: ${spec.objectType} level=$level → " +
      s"slot[level ${(level + 1).floor}] + slot[level ${level.floor} caps coverage=$coverage]"
    )
    List(
      FractionalOp(MeshFactory.createUpload(nextLevelSpec), baseMaterial),
      FractionalOp(
        MeshFactory.createUpload(currentLevelSpec, holeCaps = true),
        baseMaterial,
        isHoleCaps = true,
        coverage = coverage
      )
    )

  private final case class FractionalOp(
    plan: MeshUploadPlan,
    material: menger.common.Material,
    isHoleCaps: Boolean = false,
    coverage: Float = 1f
  )

  private def setCoverage(renderer: OptiXRenderer, id: InstanceId, coverage: Float): Unit =
    if coverage < 1f then
      val result = renderer.setInstanceCoverage(InstanceId.raw(id), coverage)
      if result != 0 then sys.error(s"setInstanceCoverage($id, $coverage) failed with $result")

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    // TD-5 resolution (Sprint 18.1): each spec gets its own mesh + GAS via per-spec
    // setTriangleMesh + addTriangleMeshInstance, so distinct triangle-mesh types coexist
    // naturally in the IAS. Each 4D spec is projected with its own parameters too (per-mesh
    // `setProjectedMesh`/CPU projection), so their projections may differ (menger#52).
    val t1 = spec1.objectType.toLowerCase
    val t2 = spec2.objectType.toLowerCase
    (!ObjectType.isSponge(t1) || spec1.level.isDefined) &&
      (!ObjectType.isSponge(t2) || spec2.level.isDefined) &&
      (!ObjectType.is4DSponge(t1) || spec1.level.isDefined) &&
      (!ObjectType.is4DSponge(t2) || spec2.level.isDefined)

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
  /** The hole caps of a fractional sponge fade out as the level rises: instance coverage
    * 1 - fractional part, with the material untouched. Not alpha: a refractive material reads
    * alpha as absorption, so glass and film caps looked the same at every level (menger#56).
    * Shared by the build and by in-place animation updates (TrackedMesh4D). */
  def holeCapsCoverage(level: Float): Float = 1f - (level - level.floor)

  /** Leaf mesh, recursion level, material and coverage of each recursive-IAS sponge instance,
    * in the order they are added. A fractional level n.f adds level n+1 on the plain cube and
    * level n on the cube's hole caps at coverage 1 - f, so only the new holes fade in instead of the
    * whole coarse level lying over the fine one (usability review 2026-09, F35 / menger#55).
    * Caps also sit on the leaf cubes' shared inner faces; they show, fading, inside the new
    * tunnels. */
  def recursiveIASInstances(
    level: Float, cube: menger.common.TriangleMeshData, material: menger.common.Material
  ): List[(menger.common.TriangleMeshData, Int, menger.common.Material, Float)] =
    if level == level.floor then List((cube, level.toInt, material, 1f))
    else List(
      (cube, level.floor.toInt + 1, material, 1f),
      (menger.objects.HoleCaps.of(cube), level.floor.toInt, material, holeCapsCoverage(level))
    )

  /** Instance transform of a (non-recursive-IAS) triangle mesh: rotation + position; size is
    * baked into the mesh. Shared by the build and by in-place moves (TrackedMesh4D). */
  def instanceTransform(spec: ObjectSpec): Array[Float] =
    TransformUtil.createEulerRotationScaleTranslation(
      spec.rotX, spec.rotY, spec.rotZ, 1f, spec.x, spec.y, spec.z
    )
