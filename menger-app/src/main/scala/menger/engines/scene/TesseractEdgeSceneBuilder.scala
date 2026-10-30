package menger.engines.scene

import scala.collection.mutable
import scala.util.Try

import io.github.lene.optix.MeshSlot
import io.github.lene.optix.OptiXRenderer
import io.github.lene.optix.TextureIndex
import menger.common.Material
import menger.common.ProfilingConfig
import menger.common.TransformUtil
import menger.common.Vector
import menger.common.x
import menger.common.y
import menger.common.z
import menger.objects.higher_d.Mesh4D
import menger.objects.higher_d.Projection
import menger.objects.higher_d.Rotation
import menger.spec.Degrees
import menger.spec.ObjectSpec
import menger.spec.ObjectType
import menger.spec.Projection4DSpec

/**
 * Scene builder for 4D hypercube objects with cylinder edge rendering.
 *
 * Renders 4D hypercube objects (tesseract, tesseract-sponge, tesseract-sponge-2) with:
 * - Faces rendered as triangle meshes (existing functionality)
 * - Edges rendered as cylinders (new functionality)
 *
 * This builder is used when hypercube objects have edge rendering parameters
 * (edge-radius, edge-material, edge-color, edge-emission).
 *
 * Key characteristics:
 * - Projects 4D edges to 3D using the same rotation and projection as faces
 * - Creates cylinder instances for each edge of the 4D mesh
 * - Supports custom edge materials and emission for glowing edges
 * - All hypercubes must have same 4D projection parameters (inherited constraint)
 *
 * Edge counts:
 * - Tesseract: 32 edges
 * - TesseractSponge level 1: 1,152 edges
 * - TesseractSponge2 level 1: 384 edges
 */
class TesseractEdgeSceneBuilder(
  textureDir: String,
  // Receives, per spec index, what the interactive rotation fast path needs to move this
  // spec's geometry in place (TesseractEdgeSceneBuilder.updateProjection).
  recorder: (Int, TesseractEdgeSceneBuilder.EdgeTrack) => Unit = (_, _) => ()
)(using profilingConfig: ProfilingConfig)
  extends SceneBuilder:
  import TesseractEdgeSceneBuilder.EdgeTrack
  import TesseractEdgeSceneBuilder.FaceTrack

  // Default edge material if none specified
  private val defaultEdgeMaterial = Material.Film

  private[scene] def isClippedByEyeW(rotated: Vector[4], eyeW: Float): Boolean =
    TesseractEdgeSceneBuilder.isClippedByEyeW(rotated, eyeW)

  /**
   * Calculate exact number of instances needed (meshes + edge cylinders).
   * Generates actual meshes to determine precise edge counts.
   */
  override def calculateRequiredInstances(specs: List[ObjectSpec]): Int =
    specs.foldLeft(0) { (total, spec) =>
      // Too large to build just to count its edges: the lower bound already exceeds any
      // instance limit (menger#53).
      val atLeast = TesseractEdgeSceneBuilder.minEdgeInstancesOf(spec)
      val instances =
        if atLeast > 0 then math.min(atLeast, Int.MaxValue.toLong).toInt
        else
          val meshInstances = 1  // The main mesh
          val edgeInstances = if spec.hasEdgeRendering then
            TesseractEdgeSceneBuilder.edgeLevels(spec).map((levelSpec, _) =>
              levelEdges(levelSpec).size
            ).sum
          else
            0
          meshInstances + edgeInstances
      total + instances
    }

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then
      Left("Object specs list cannot be empty")
    else if !specs.forall(s => ObjectType.isProjected4D(s.objectType)) then
      Left("TesseractEdgeSceneBuilder only supports 4D projected types")
    else if !specs.forall(_.hasEdgeRendering) then
      Left("TesseractEdgeSceneBuilder requires edge rendering parameters on all specs")
    else
      // Each spec gets its own face mesh and edge cylinders, projected with its own
      // `projection4D`, so 4D objects with different projections coexist (usability review
      // 2026-09, session 2, F44/menger#52).
      // A sponge too big to build just to count its edges is rejected from a lower bound
      // first (menger#53); everything else gets the exact count from its mesh.
      val atLeast = specs.map(TesseractEdgeSceneBuilder.minEdgeInstancesOf).sum
      lazy val requiredInstances = calculateRequiredInstances(specs)
      if atLeast > maxInstances then
        Left(
          s"Scene requires at least $atLeast instances (one cylinder per edge) but the limit " +
          s"is $maxInstances. Edge-rendered tesseract sponges fit up to level 2; use a lower " +
          "level or drop `edgeRadius`."
        )
      else if requiredInstances > maxInstances then
        val recommended = Math.min(requiredInstances * 2, menger.common.Const.maxInstancesLimit)
        Left(
          s"Scene requires $requiredInstances instances (including edge cylinders) but limit is $maxInstances. " +
          s"Recommendation: Add --max-instances $recommended to your command. " +
          "Note: Edge rendering creates one cylinder per edge (varies by object type and level)."
        )
      else
        Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] = Try:
    logger.debug(
      s"Building scene with edge rendering: ${specs.length} ${specs.head.objectType}(s)")

    // Only when the renderer can't hold the scene: a reinitialize tears the whole native renderer
    // down and rebuilds it (~46 ms), and this runs on every rebuild -- for an edge-rendered
    // polytope that used to mean every step of an interactive rotation (usability review
    // 2026-09, F22). It also discards lights and render settings, which callers reapply.
    if maxInstances > renderer.maxInstances then
      logger.debug(s"Reinitializing renderer with maxInstances=$maxInstances")
      renderer.reinitialize(maxInstances)

    // Load textures
    val textureIndices = TextureManager.loadTextures(specs, renderer, textureDir)

    // Add instances for each tesseract
    specs.zipWithIndex.foreach { case (spec, specIdx) =>
      val position = Vector[3](spec.x, spec.y, spec.z)

      val hasFaceMaterial = spec.material.isDefined
      val hasEdgeMaterial = spec.edgeMaterial.isDefined

      // Only add face mesh instance if face material is specified (not just edge material)
      val faces =
        if hasFaceMaterial then addFaces(spec, position, textureIndices, renderer)
        else FaceTrack.NoFaces

      // Add edge cylinder instances if edge material or edge radius specified
      val (edges, cylinderIds) =
        if hasEdgeMaterial || spec.edgeRadius.isDefined then addEdgeCylinders(spec, renderer)
        else
          logger.debug("Skipping edge cylinders (no edge material/radius specified)")
          (IndexedSeq.empty, IndexedSeq.empty)
      recorder(specIdx, EdgeTrack(faces, edges, cylinderIds))
    }

  private def addFaces(
    spec: ObjectSpec,
    position: Vector[3],
    textureIndices: Map[String, TextureIndex],
    renderer: OptiXRenderer
  ): FaceTrack =
    // Each spec's own mesh: addTriangleMeshInstance instances the most recently set one.
    // One shared mesh built from `specs.head` gave a tesseract next to a sponge the
    // sponge's shape and size (usability review 2026-09, F27).
    // ponytail: one mesh per spec, cache by (type, level, size, projection) if many
    // identical 4D objects ever make this slow.
    val faceTrack = uploadFaces(spec, renderer)
    val faceMaterial = MaterialExtractor.extract(spec)
    val textureIndex =
      spec.imageTextureKey.flatMap(textureIndices.get).getOrElse(TextureIndex.NoTexture)

    val faceInstanceId =
      if spec.rotX == 0f && spec.rotY == 0f && spec.rotZ == 0f then
        renderer.addTriangleMeshInstance(position, faceMaterial, textureIndex)
      else
        val transform = TransformUtil.createEulerRotationScaleTranslation(
          spec.rotX, spec.rotY, spec.rotZ, 1f, spec.x, spec.y, spec.z
        )
        renderer.addTriangleMeshInstance(transform, faceMaterial, textureIndex)

    // The shared texture/procedural/object-frame step every other builder runs (usability
    // review 2026-10, session 4, F97: faces had no procedural colour on this path).
    applyInstanceTextures(faceInstanceId, spec, textureIndices, renderer)
    logger.debug(s"Added tesseract face mesh instance ${faceInstanceId.value} at ($position)")
    faceTrack

  /** 4D faces go through the GPU projection path wherever it applies -- the same upload
    * `TriangleMeshSceneBuilder` uses for these types -- so the rotation fast path can re-project
    * them in place (`updateMesh4DProjection`). Fractional-level sponges keep the CPU mesh (their
    * two-level blend is built on the CPU) and so rebuild on rotation. */
  private def uploadFaces(spec: ObjectSpec, renderer: OptiXRenderer): FaceTrack =
    val integralLevel = spec.level.forall(l => l == l.floor)
    if integralLevel then
      MeshFactory.createUpload(spec) match
        case MeshUploadPlan.Gpu4D(faces4D, vertsPerFace, proj) =>
          FaceTrack.Gpu(renderer.setProjectedMesh(
            faces4D, vertsPerFace, uvs = null, // scalafix:ok DisableSyntax.null
            eyeW = proj.eyeW, screenW = proj.screenW,
            rotXW = proj.rotXW.value, rotYW = proj.rotYW.value, rotZW = proj.rotZW.value,
            centerX = 0f, centerY = 0f, centerZ = 0f
          ))
        case MeshUploadPlan.Cpu(data) =>
          renderer.addTriangleMesh(data)
          FaceTrack.Cpu
    else
      renderer.addTriangleMesh(MeshFactory.create(spec))
      FaceTrack.Cpu

  /**
   * Add cylinder instances for all edges of a 4D hypercube mesh.
   *
   * Edge counts vary by mesh type:
   * - Tesseract: 32 edges
   * - TesseractSponge: grows exponentially with level
   * - TesseractSponge2: grows exponentially with level
   *
   * Each edge is projected from 4D to 3D using the same rotation and projection as the faces.
   */
  private def addEdgeCylinders(
    spec: ObjectSpec,
    renderer: OptiXRenderer
  ): (IndexedSeq[(Vector[4], Vector[4])], IndexedSeq[Option[InstanceId]]) =
    val edgeRadius = spec.edgeRadius.getOrElse(TesseractEdgeSceneBuilder.DefaultEdgeRadius)
    val edgeMaterial = spec.edgeMaterial.getOrElse(defaultEdgeMaterial)
    // A fractional level n.f draws the edges of level n fading out and those of level n+1 fading
    // in (usability review 2026-10, session 4, F94): they used to be built from the integer
    // part, so the edges of a level animation stepped at each integer.
    val groups = TesseractEdgeSceneBuilder.edgeLevels(spec).map { (levelSpec, keep) =>
      val groupEdges = levelEdges(levelSpec)
      val material = TesseractEdgeSceneBuilder.fadedEdgeMaterial(edgeMaterial, keep)
      (groupEdges, TesseractEdgeSceneBuilder.edgeEndpoints(spec, groupEdges), material)
    }
    val edges = groups.flatMap(_._1).toIndexedSeq

    // One cylinder per edge that isn't clipped by the eye_w plane (None: clipped, no cylinder).
    val cylinderIds = groups.flatMap { (_, endpoints, material) =>
      endpoints.map(_.map { case (p0, p1) =>
        val cylinderId = renderer.addCylinderInstance(p0, p1, edgeRadius, material)
        logger.trace(s"Added edge cylinder ${cylinderId.value} from $p0 to $p1")
        cylinderId
      })
    }.toIndexedSeq

    logger.debug(s"Added ${cylinderIds.count(_.isDefined)} of ${edges.size} edge cylinders for " +
      s"${spec.objectType} at (${spec.x}, ${spec.y}, ${spec.z})")
    (edges, cylinderIds)

  // Counting, validating and building all ask for the same levels' edges; each meant a fresh 4D
  // mesh and dedup, three per rebuild (menger#129). Keyed without the view, which the edges do
  // not depend on. A builder serves one build, so the memo lives as long as that build.
  private val edgesByLevel = mutable.Map.empty[ObjectSpec, IndexedSeq[(Vector[4], Vector[4])]]

  private def levelEdges(levelSpec: ObjectSpec): IndexedSeq[(Vector[4], Vector[4])] =
    edgesByLevel.getOrElseUpdate(
      levelSpec.withoutView, extractEdges(createMesh4D(levelSpec)).toIndexedSeq
    )

  // The same w-scaled mesh as the faces (menger#65), not a second copy of the type switch.
  private def createMesh4D(spec: ObjectSpec): Mesh4D =
    MeshFactory.renderMesh4D(spec.copy(objectType = spec.objectType.toLowerCase)).getOrElse(
      sys.error(s"Unsupported 4D type for edge rendering (or no level): ${spec.objectType}")
    )

  /**
   * Extract all unique edges from a 4D mesh.
   *
   * Edges are extracted from the quad faces by taking each edge of each face
   * and deduplicating using canonical ordering.
   *
   * @param mesh4D The 4D mesh to extract edges from
   * @return Set of unique edges as (start, end) vertex pairs
   */
  private def extractEdges(mesh4D: Mesh4D): Seq[(Vector[4], Vector[4])] =
    mesh4D.faces.flatMap { face =>
      val vpf = face.vertsPerFace
      val verts = (0 until vpf).map(face(_))
      verts.zip(verts.tail :+ verts.head)
    }.map { case (v1, v2) =>
      if compareVectors(v1, v2) < 0 then (v1, v2) else (v2, v1)
    }.distinct

  /**
   * Compare two 4D vectors lexicographically.
   * Returns negative if v1 < v2, positive if v1 > v2, zero if equal.
   */
  private def compareVectors(v1: Vector[4], v2: Vector[4]): Int =
    (0 until 4).map { i =>
      v1(i).compare(v2(i))
    }.find(_ != 0).getOrElse(0)

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    // Both must be 4D projected types; their projections may differ (menger#52).
    ObjectType.isProjected4D(spec1.objectType) && ObjectType.isProjected4D(spec2.objectType)

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    // Calculate total instances: 1 face mesh instance + N edge cylinder instances per object
    // Edge counts vary by type:
    // - Tesseract: 32 edges
    // - TesseractSponge level 1: ~1,152 edges
    // - TesseractSponge2 level 1: ~384 edges
    specs.map { spec =>
      val edgeCount = estimateEdgeCount(spec)
      1L + edgeCount  // 1 face mesh + N edge cylinders
    }.sum

  /** Estimate edge count for quick instance budget checks (no mesh instantiation). */
  private def estimateEdgeCount(spec: ObjectSpec): Long =
    spec.objectType.toLowerCase match
      case "tesseract"                                     => 32L
      case "pentachoron"                                   => 10L
      case "16-cell"                                       => 24L
      case "24-cell"                                       => 96L
      case "600-cell"                                      => 720L
      case "120-cell"                                      => 1200L
      case "tesseract-sponge" | "tesseract-sponge-volume" =>
        val level = spec.level.map(_.toInt).getOrElse(0)
        import menger.objects.higher_d.TesseractSpongeMesh
        TesseractSpongeMesh.estimatedFaces(level) * 2
      case "tesseract-sponge-2" | "tesseract-sponge-surface" =>
        val level = spec.level.map(_.toInt).getOrElse(0)
        import menger.objects.higher_d.TesseractSponge2Mesh
        TesseractSponge2Mesh.estimatedFaces(level) * 2
      case _ => 32L

object TesseractEdgeSceneBuilder:

  val DefaultEdgeRadius = 0.02f

  /** Measured edge counts at level 2, and the smallest per-level growth seen from level 0 up
    * (volume 24x then 36x, surface 21x then 16.1x; surface rounded down to 12 for margin). */
  private val Level2Edges = Map("volume" -> 27648L, "surface" -> 10848L)
  private val MinGrowthPerLevel = Map("volume" -> 24L, "surface" -> 12L)
  private val SmallestUncountedLevel = 3

  /** Instances an edge-rendered tesseract sponge needs at least -- without building its mesh,
    * which from level 3 up is too large to build just to count edges (usability review
    * 2026-09, session 2, F33/menger#53). 0 for everything that is cheap to count exactly. */
  def minEdgeInstances(spec: ObjectSpec): Long =
    val variant = spec.objectType.toLowerCase match
      case "tesseract-sponge" | "tesseract-sponge-volume" => Some("volume")
      case "tesseract-sponge-2" | "tesseract-sponge-surface" => Some("surface")
      case _ => None
    val level = spec.level.map(_.floor.toInt).getOrElse(0)
    variant.filter(_ => spec.hasEdgeRendering && level >= SmallestUncountedLevel) match
      case Some(v) => 1L + Level2Edges(v) * math.pow(MinGrowthPerLevel(v).toDouble, level - 2).toLong
      case None => 0L

  /** Highest level whose edges cross-fade with the next one's: level 3 needs 130177 cylinders,
    * more than the instance limit, so from 2.x the edges stay those of the integer part. */
  private val MaxCrossFadeLevel = 2

  /** The integer-level specs whose edges are drawn for `spec`, each with the share it keeps:
    * the spec itself (1) at an integer level; level n (1 - f) and level n+1 (f) at n.f while
    * n+1 fits `MaxCrossFadeLevel`; else level n alone. */
  def edgeLevels(spec: ObjectSpec): List[(ObjectSpec, Float)] =
    spec.level match
      case Some(level) if level != level.floor =>
        val lower = spec.copy(level = Some(level.floor))
        val fraction = level - level.floor
        if level.floor + 1 <= MaxCrossFadeLevel then
          List((lower, 1f - fraction), (spec.copy(level = Some(level.floor + 1f)), fraction))
        else List((lower, 1f))
      case _ => List((spec, 1f))

  /** `minEdgeInstances` over every level whose edges are drawn for `spec`. */
  def minEdgeInstancesOf(spec: ObjectSpec): Long =
    edgeLevels(spec).map((levelSpec, _) => minEdgeInstances(levelSpec)).sum

  /** An edge material that keeps `keep` of its opacity and, when refractive, of its refraction
    * (alpha and IOR above 1 scaled): the rule `TriangleMeshSceneBuilder.fadingMaterial` uses for
    * a fading refractive cap, in the same direction (decided by user 2026-10-13: "decrease
    * opacity to 0 and ior to 1 with increasing fractional level"). */
  def fadedEdgeMaterial(material: Material, keep: Float): Material =
    if keep >= 1f then material
    else
      material.copy(
        color = material.color.copy(a = material.color.a * keep),
        ior = if material.ior > 1f then 1f + (material.ior - 1f) * keep else material.ior
      )

  // Matches the epsilon the 4D CUDA shaders use for the same eye_w clip (e.g. hit_menger4d.cu).
  private val EyeWClipEpsilon = 1e-6f

  private[scene] def isClippedByEyeW(rotated: Vector[4], eyeW: Float): Boolean =
    rotated(3) >= eyeW - EyeWClipEpsilon

  /** How a spec's faces were uploaded: re-projectable in place on the GPU, CPU-projected
    * (a rotation has to rebuild), or none (edges only). */
  enum FaceTrack:
    case NoFaces
    case Gpu(meshSlot: MeshSlot)
    case Cpu

  /** What the interactive rotation fast path needs to move one spec's geometry in place.
    * `cylinderIds` is aligned with `edges`: the cylinder drawn for that edge, or None where the
    * edge was clipped by the eye_w plane at build time. */
  final case class EdgeTrack(
    faces: FaceTrack,
    edges: IndexedSeq[(Vector[4], Vector[4])],
    cylinderIds: IndexedSeq[Option[InstanceId]]
  )

  /** 3D endpoints of each 4D edge under the spec's rotation and projection, offset to its
    * position; None for an edge with an endpoint at or behind the eye_w projection plane.
    * That mirrors the clip every 4D CUDA closest-hit shader applies (e.g. hit_menger4d.cu's
    * `rot.w >= m.eye_w - 1e-6f`): without it, Projection.apply's
    * `(eyeW - screenW) / (eyeW - point(3))` denominator approaches or crosses zero once a
    * vertex's w reaches eyeW, giving a non-finite or exploded endpoint. Rotation preserves a
    * vertex's 4D norm, so at the CLI defaults (eyeW=3.0, size ~0.8-1.5) no rotation reaches
    * the clip; it matters once --eye-w is brought close to --size (Sprint 36 H1.5). */
  def edgeEndpoints(
    spec: ObjectSpec,
    edges: IndexedSeq[(Vector[4], Vector[4])]
  ): IndexedSeq[Option[(Vector[3], Vector[3])]] =
    val proj4D = spec.projection4D.getOrElse(Projection4DSpec.default)
    val rotation: Rotation =
      if proj4D.rotXW == Degrees.Zero && proj4D.rotYW == Degrees.Zero &&
        proj4D.rotZW == Degrees.Zero then Rotation.identity
      else Rotation(
        proj4D.rotXW.value, proj4D.rotYW.value, proj4D.rotZW.value, Vector[4](0f, 0f, 0f, 0f)
      )
    val projection = Projection(proj4D.eyeW, proj4D.screenW)
    edges.map { case (v0, v1) =>
      val r0 = rotation(v0)
      val r1 = rotation(v1)
      if isClippedByEyeW(r0, proj4D.eyeW) || isClippedByEyeW(r1, proj4D.eyeW) then None
      else
        val p0 = projection(r0)
        val p1 = projection(r1)
        Some((
          Vector[3](p0.x + spec.x, p0.y + spec.y, p0.z + spec.z),
          Vector[3](p1.x + spec.x, p1.y + spec.y, p1.z + spec.z)
        ))
    }

  /** Interactive 4D rotation fast path: moves the already-built edge cylinders (and
    * GPU-projected faces) of every spec whose projection changed, instead of rebuilding the
    * scene (usability review 2026-09, F22). Returns false -- nothing changed, the caller
    * rebuilds -- unless at most projections changed, every changed spec's faces are
    * re-projectable, and the set of edges clipped by the eye_w plane is the same as when the
    * cylinders were built (a newly clipped or unclipped edge needs a different cylinder
    * count). Everything is checked before anything is moved. */
  def updateProjection(
    prevSpecs: List[ObjectSpec],
    newSpecs: List[ObjectSpec],
    tracks: IndexedSeq[EdgeTrack],
    renderer: OptiXRenderer
  ): Boolean =
    if prevSpecs.size != tracks.size || newSpecs.size != tracks.size
      || !menger.engines.WithAnimation.specsDifferAtMostIn4DProjection(prevSpecs, newSpecs)
    then false
    else
      val changed = prevSpecs.lazyZip(newSpecs).lazyZip(tracks).collect {
        case (prev, next, track) if prev.projection4D != next.projection4D => (next, track)
      }.toIndexedSeq
      val planned = changed.map { case (spec, track) =>
        (spec, track, edgeEndpoints(spec, track.edges))
      }
      val movable = planned.forall { case (_, track, endpoints) =>
        track.faces != FaceTrack.Cpu &&
          endpoints.map(_.isDefined) == track.cylinderIds.map(_.isDefined)
      }
      if !movable then false
      else
        planned.foreach { case (spec, track, endpoints) =>
          val moves = track.cylinderIds.lazyZip(endpoints).collect {
            case (Some(id), Some((p0, p1))) => (id, p0, p1)
          }.toIndexedSeq
          if moves.nonEmpty then
            val radius = spec.edgeRadius.getOrElse(DefaultEdgeRadius)
            renderer.updateCylinderInstances(
              moves.map(_._1).toArray,
              moves.flatMap { case (_, p0, _) => Seq(p0.x, p0.y, p0.z) }.toArray,
              moves.flatMap { case (_, _, p1) => Seq(p1.x, p1.y, p1.z) }.toArray,
              Array.fill(moves.size)(radius)
            )
          track.faces match
            case FaceTrack.Gpu(slot) =>
              val proj = spec.projection4D.getOrElse(Projection4DSpec.default)
              renderer.updateMesh4DProjection(
                slot, eyeW = proj.eyeW, screenW = proj.screenW,
                rotXW = proj.rotXW.value, rotYW = proj.rotYW.value, rotZW = proj.rotZW.value
              )
            case _ => ()
        }
        true
