package menger.engines.scene

import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.Material
import menger.common.ObjectType
import menger.common.ProfilingConfig
import menger.common.TransformUtil
import menger.common.Vector
import menger.common.x
import menger.common.y
import menger.common.z
import menger.objects.higher_d.Hecatonicosachoron
import menger.objects.higher_d.Hexacosichoron
import menger.objects.higher_d.Hexadecachoron
import menger.objects.higher_d.Icositetrachoron
import menger.objects.higher_d.Mesh4D
import menger.objects.higher_d.Pentachoron
import menger.objects.higher_d.Projection
import menger.objects.higher_d.Rotation
import menger.objects.higher_d.Tesseract
import menger.objects.higher_d.TesseractSponge
import menger.objects.higher_d.TesseractSponge2

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
      val meshInstances = 1  // The main mesh
      val edgeInstances = if spec.hasEdgeRendering then
        extractEdges(createMesh4D(spec)).size
      else
        0
      total + meshInstances + edgeInstances
    }

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then
      Left("Object specs list cannot be empty")
    else if !specs.forall(s => ObjectType.isProjected4D(s.objectType)) then
      Left("TesseractEdgeSceneBuilder only supports 4D projected types")
    else if !specs.forall(_.hasEdgeRendering) then
      Left("TesseractEdgeSceneBuilder requires edge rendering parameters on all specs")
    else
      // Check compatibility - all specs must have same 4D projection params
      val firstSpec = specs.head
      specs.find(!isCompatible(_, firstSpec)) match
        case Some(incompatible) =>
          Left("Incompatible 4D projection parameters between 4D objects. " +
            "All 4D objects must have matching projection parameters for shared mesh rendering.")
        case None =>
          // Calculate actual required instances by generating meshes
          val requiredInstances = calculateRequiredInstances(specs)

          if requiredInstances > maxInstances then
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
    textureIndices: Map[String, Int],
    renderer: OptiXRenderer
  ): FaceTrack =
    // Each spec's own mesh: addTriangleMeshInstance instances the most recently set one.
    // One shared mesh built from `specs.head` gave a tesseract next to a sponge the
    // sponge's shape and size (usability review 2026-09, F27).
    // ponytail: one mesh per spec, cache by (type, level, size, projection) if many
    // identical 4D objects ever make this slow.
    val faceTrack = uploadFaces(spec, renderer)
    val faceMaterial = MaterialExtractor.extract(spec)
    val textureIndex = spec.imageTextureKey.flatMap(textureIndices.get).getOrElse(-1)

    val faceInstanceId =
      if spec.rotX == 0f && spec.rotY == 0f && spec.rotZ == 0f then
        renderer.addTriangleMeshInstance(position, faceMaterial, textureIndex)
      else
        val transform = TransformUtil.createEulerRotationScaleTranslation(
          spec.rotX, spec.rotY, spec.rotZ, 1f, spec.x, spec.y, spec.z
        )
        renderer.addTriangleMeshInstance(transform, faceMaterial, textureIndex)

    val validFaceInstanceId = requireInstanceId(
      faceInstanceId,
      s"tesseract face mesh instance at ($position)"
    )
    logger.debug(s"Added tesseract face mesh instance $validFaceInstanceId at ($position)")
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
            rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW,
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
  ): (IndexedSeq[(Vector[4], Vector[4])], IndexedSeq[Option[Int]]) =
    val edgeRadius = spec.edgeRadius.getOrElse(TesseractEdgeSceneBuilder.DefaultEdgeRadius)
    val edgeMaterial = spec.edgeMaterial.getOrElse(defaultEdgeMaterial)
    val edges = extractEdges(createMesh4D(spec)).toIndexedSeq
    val endpoints = TesseractEdgeSceneBuilder.edgeEndpoints(spec, edges)

    // One cylinder per edge that isn't clipped by the eye_w plane (None: clipped, no cylinder).
    val cylinderIds = endpoints.map(_.map { case (p0, p1) =>
      val cylinderId = requireInstanceId(
        renderer.addCylinderInstance(p0, p1, edgeRadius, edgeMaterial),
        s"edge cylinder from $p0 to $p1"
      )
      logger.trace(s"Added edge cylinder $cylinderId from $p0 to $p1")
      InstanceId.raw(cylinderId)
    })

    logger.debug(s"Added ${cylinderIds.count(_.isDefined)} of ${edges.size} edge cylinders for " +
      s"${spec.objectType} at (${spec.x}, ${spec.y}, ${spec.z})")
    (edges, cylinderIds)

  private def createMesh4D(spec: ObjectSpec): Mesh4D =
    spec.objectType.toLowerCase match
      case "tesseract" =>
        Tesseract(size = spec.size)
      case "tesseract-sponge" | "tesseract-sponge-volume" =>
        require(spec.level.isDefined, "tesseract-sponge requires level parameter")
        TesseractSponge(spec.level.get, spec.size)
      case "tesseract-sponge-2" | "tesseract-sponge-surface" =>
        require(spec.level.isDefined, "tesseract-sponge-2 requires level parameter")
        TesseractSponge2(spec.level.get, spec.size)
      case "pentachoron" =>
        Pentachoron(size = spec.size)
      case "16-cell" =>
        Hexadecachoron(size = spec.size)
      case "24-cell" =>
        Icositetrachoron(size = spec.size)
      case "600-cell" =>
        Hexacosichoron(size = spec.size)
      case "120-cell" =>
        Hecatonicosachoron(size = spec.size)
      case other =>
        sys.error(s"Unsupported 4D type for edge rendering: $other")

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
    // Both must be 4D projected types
    if !ObjectType.isProjected4D(spec1.objectType) || !ObjectType.isProjected4D(spec2.objectType) then
      false
    else
      // Must have same 4D projection params (for shared mesh geometry)
      (spec1.projection4D, spec2.projection4D) match
        case (Some(p1), Some(p2)) =>
          p1.eyeW == p2.eyeW && p1.screenW == p2.screenW &&
          p1.rotXW == p2.rotXW && p1.rotYW == p2.rotYW && p1.rotZW == p2.rotZW
        case (None, None) => true  // Both using defaults
        case _ => false

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

  // Matches the epsilon the 4D CUDA shaders use for the same eye_w clip (e.g. hit_menger4d.cu).
  private val EyeWClipEpsilon = 1e-6f

  private[scene] def isClippedByEyeW(rotated: Vector[4], eyeW: Float): Boolean =
    rotated(3) >= eyeW - EyeWClipEpsilon

  /** How a spec's faces were uploaded: re-projectable in place on the GPU, CPU-projected
    * (a rotation has to rebuild), or none (edges only). */
  enum FaceTrack:
    case NoFaces
    case Gpu(meshSlot: Int)
    case Cpu

  /** What the interactive rotation fast path needs to move one spec's geometry in place.
    * `cylinderIds` is aligned with `edges`: the cylinder drawn for that edge, or None where the
    * edge was clipped by the eye_w plane at build time. */
  final case class EdgeTrack(
    faces: FaceTrack,
    edges: IndexedSeq[(Vector[4], Vector[4])],
    cylinderIds: IndexedSeq[Option[Int]]
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
      if proj4D.rotXW == 0f && proj4D.rotYW == 0f && proj4D.rotZW == 0f then Rotation.identity
      else Rotation(proj4D.rotXW, proj4D.rotYW, proj4D.rotZW, Vector[4](0f, 0f, 0f, 0f))
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
    * rebuilds -- unless only projections changed, every changed spec's faces are
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
      || !menger.engines.WithAnimation.specsDifferOnlyIn4DProjection(prevSpecs, newSpecs)
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
                rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW
              )
            case _ => ()
        }
        true
