package menger.engines.scene

import scala.util.Try

import com.badlogic.gdx.math.Vector3
import menger.ObjectSpec
import menger.ProfilingConfig
import menger.Projection4DSpec
import menger.common.ObjectType
import menger.common.TransformUtil
import menger.common.Vector
import menger.objects.higher_d.Mesh4D
import menger.objects.higher_d.Projection
import menger.objects.higher_d.Rotation
import menger.objects.higher_d.Tesseract
import menger.objects.higher_d.TesseractSponge
import menger.objects.higher_d.TesseractSponge2
import menger.optix.Material
import menger.optix.OptiXRenderer

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
        // Generate 4D mesh to get actual edge count
        val mesh4D: Mesh4D = spec.objectType.toLowerCase match
          case "tesseract" =>
            Tesseract(size = spec.size)
          case "tesseract-sponge" =>
            require(spec.level.isDefined, "tesseract-sponge requires level parameter")
            TesseractSponge(spec.level.get)
          case "tesseract-sponge-2" =>
            require(spec.level.isDefined, "tesseract-sponge-2 requires level parameter")
            TesseractSponge2(spec.level.get, spec.size)
          case other =>
            require(false, s"Unknown 4D object type: $other")
            Tesseract(size = spec.size)  // Never reached, but needed for type checker

        val edges = extractEdges(mesh4D)
        edges.size
      else
        0
      total + meshInstances + edgeInstances
    }

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then
      Left("Object specs list cannot be empty")
    else if !specs.forall(s => ObjectType.isProjected4D(s.objectType)) then
      Left("TesseractEdgeSceneBuilder only supports 4D projected types (tesseract, tesseract-sponge, tesseract-sponge-2)")
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
        val textureIndex = spec.texture.flatMap(textureIndices.get).getOrElse(-1)

        val faceInstanceId =
          if spec.rotX == 0f && spec.rotY == 0f && spec.rotZ == 0f then
            renderer.addTriangleMeshInstance(position, faceMaterial, textureIndex)
          else
            val transform = TransformUtil.createEulerRotationScaleTranslation(
              spec.rotX, spec.rotY, spec.rotZ, 1f, spec.x, spec.y, spec.z
            )
            renderer.addTriangleMeshInstance(transform, faceMaterial, textureIndex)

        faceInstanceId match
          case Some(id) =>
            logger.debug(s"Added tesseract face mesh instance $id at ($position)")
          case None =>
            logger.error(s"Failed to add tesseract face mesh instance at ($position)")

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

    // Create the appropriate 4D mesh based on object type
    val mesh4D: Mesh4D = spec.objectType.toLowerCase match
      case "tesseract" =>
        Tesseract(size = spec.size)
      case "tesseract-sponge" =>
        require(spec.level.isDefined, "tesseract-sponge requires level parameter")
        TesseractSponge(spec.level.get)
      case "tesseract-sponge-2" =>
        require(spec.level.isDefined, "tesseract-sponge-2 requires level parameter")
        TesseractSponge2(spec.level.get, spec.size)
      case other =>
        require(false, s"Unsupported hypercube type for edge rendering: $other")
        // Never reached due to require, but needed for type checker
        Tesseract(size = spec.size)

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
    val edgeCount = edges.count { case (v0_4d, v1_4d) =>
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
      renderer.addCylinderInstance(p0, p1, edgeRadius, edgeMaterial) match
        case Some(id) =>
          logger.trace(s"Added edge cylinder $id from $p0 to $p1")
          true
        case None =>
          logger.warn(s"Failed to add edge cylinder from $p0 to $p1")
          false
    }

    logger.debug(s"Added $edgeCount edge cylinders for ${spec.objectType} at (${spec.x}, ${spec.y}, ${spec.z})")

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
      // Each quad face has 4 edges: (a,b), (b,c), (c,d), (d,a)
      val vertices = Seq(face.a, face.b, face.c, face.d)
      vertices.zip(vertices.tail :+ vertices.head)
    }.map { case (v1, v2) =>
      // Use canonical ordering to deduplicate edges (smaller vector first)
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

  /**
   * Estimate the number of edges for a given hypercube spec.
   * Uses the face count as a proxy (each quad face has 4 edges, but edges are shared).
   */
  private def estimateEdgeCount(spec: ObjectSpec): Long =
    spec.objectType.toLowerCase match
      case "tesseract" =>
        32L  // Known constant
      case "tesseract-sponge" =>
        val level = spec.level.map(_.toInt).getOrElse(0)
        // Approximate: each face has 4 edges, edges are shared 2:1
        import menger.objects.higher_d.TesseractSpongeMesh
        TesseractSpongeMesh.estimatedFaces(level) * 2
      case "tesseract-sponge-2" =>
        val level = spec.level.map(_.toInt).getOrElse(0)
        // Approximate: each face has 4 edges, edges are shared 2:1
        import menger.objects.higher_d.TesseractSponge2Mesh
        TesseractSponge2Mesh.estimatedFaces(level) * 2
      case _ =>
        32L  // Default conservative estimate
