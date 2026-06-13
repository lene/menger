package menger.engines.scene

import scala.util.Try

import com.badlogic.gdx.math.Vector3
import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.Material
import menger.common.ObjectType
import menger.common.ProfilingConfig
import menger.common.TransformUtil
import menger.common.Vector
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
class TesseractEdgeSceneBuilder(textureDir: String)(using profilingConfig: ProfilingConfig)
  extends SceneBuilder:

  // Default edge material if none specified
  private val defaultEdgeMaterial = Material.Film

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
    logger.info(s"Building tesseract scene with edge rendering: ${specs.length} tesseracts")

    // Reinitialize renderer with correct maxInstances if needed
    if maxInstances > 64 then
      logger.debug(s"Reinitializing renderer with maxInstances=$maxInstances")
      renderer.reinitialize(maxInstances)

    // Create shared base geometry for faces
    val firstSpec = specs.head
    val mesh = MeshFactory.create(firstSpec)
    renderer.setTriangleMesh(mesh)

    // Load textures
    val textureIndices = TextureManager.loadTextures(specs, renderer, textureDir)

    // Add instances for each tesseract
    specs.foreach { spec =>
      val position = Vector[3](spec.x, spec.y, spec.z)

      val hasFaceMaterial = spec.material.isDefined
      val hasEdgeMaterial = spec.edgeMaterial.isDefined

      // Only add face mesh instance if face material is specified (not just edge material)
      if hasFaceMaterial then
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

      // Add edge cylinder instances if edge material or edge radius specified
      if hasEdgeMaterial || spec.edgeRadius.isDefined then
        addEdgeCylinders(spec, renderer)
      else
        logger.debug("Skipping edge cylinders (no edge material/radius specified)")
    }

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
  private def addEdgeCylinders(spec: ObjectSpec, renderer: OptiXRenderer): Unit =
    val proj4D = spec.projection4D.getOrElse(Projection4DSpec.default)
    val edgeRadius = spec.edgeRadius.getOrElse(0.02f)
    val edgeMaterial = spec.edgeMaterial.getOrElse(defaultEdgeMaterial)

    val mesh4D: Mesh4D = createMesh4D(spec)

    // Create rotation and projection (same as TesseractMesh)
    val rotation: Rotation =
      if proj4D.rotXW == 0f && proj4D.rotYW == 0f && proj4D.rotZW == 0f then
        Rotation.identity
      else
        Rotation(proj4D.rotXW, proj4D.rotYW, proj4D.rotZW, Vector[4](0f, 0f, 0f, 0f))

    val projection = Projection(proj4D.eyeW, proj4D.screenW)

    // Position offset for this hypercube instance
    val offset = Vector3(spec.x, spec.y, spec.z)

    // Extract edges from the 4D mesh
    val edges = extractEdges(mesh4D)

    // Project each edge and create cylinder
    edges.foreach { case (v0_4d, v1_4d) =>
      // Apply 4D rotation
      val rotatedV0 = rotation(v0_4d)
      val rotatedV1 = rotation(v1_4d)

      // Project to 3D
      val p0_3d = projection(rotatedV0)
      val p1_3d = projection(rotatedV1)

      // Apply position offset
      val p0 = Vector[3](p0_3d.x + offset.x, p0_3d.y + offset.y, p0_3d.z + offset.z)
      val p1 = Vector[3](p1_3d.x + offset.x, p1_3d.y + offset.y, p1_3d.z + offset.z)

      // Add cylinder instance
      val cylinderId = requireInstanceId(
        renderer.addCylinderInstance(p0, p1, edgeRadius, edgeMaterial),
        s"edge cylinder from $p0 to $p1"
      )
      logger.trace(s"Added edge cylinder $cylinderId from $p0 to $p1")
    }

    logger.debug(s"Added ${edges.size} edge cylinders for ${spec.objectType} at (${spec.x}, ${spec.y}, ${spec.z})")

  private def createMesh4D(spec: ObjectSpec): Mesh4D =
    spec.objectType.toLowerCase match
      case "tesseract" =>
        Tesseract(size = spec.size)
      case "tesseract-sponge" | "tesseract-sponge-volume" =>
        require(spec.level.isDefined, "tesseract-sponge requires level parameter")
        TesseractSponge(spec.level.get)
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
