package menger.engines.scene

import scala.util.Try

import com.badlogic.gdx.math.Vector3
import menger.ObjectSpec
import menger.ProfilingConfig
import menger.Projection4DSpec
import menger.common.ObjectType
import menger.common.Vector
import menger.objects.higher_d.Projection
import menger.objects.higher_d.Rotation
import menger.objects.higher_d.Tesseract
import menger.optix.Material
import menger.optix.OptiXRenderer

/**
 * Scene builder for tesseract objects with cylinder edge rendering.
 *
 * Renders tesseract (4D hypercube) objects with:
 * - Faces rendered as triangle meshes (existing functionality)
 * - Edges rendered as cylinders (new functionality)
 *
 * This builder is used when tesseract objects have edge rendering parameters
 * (edge-radius, edge-material, edge-color, edge-emission).
 *
 * Key characteristics:
 * - Projects 4D edges to 3D using the same rotation and projection as faces
 * - Creates cylinder instances for each of the 32 edges of the tesseract
 * - Supports custom edge materials and emission for glowing edges
 * - All tesseracts must have same 4D projection parameters (inherited constraint)
 */
class TesseractEdgeSceneBuilder(textureDir: String)(using profilingConfig: ProfilingConfig)
  extends SceneBuilder:

  // Default edge material if none specified
  private val defaultEdgeMaterial = Material.Film

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then
      Left("Object specs list cannot be empty")
    else if !specs.forall(s => ObjectType.isHypercube(s.objectType)) then
      Left("TesseractEdgeSceneBuilder only supports hypercube types (tesseract)")
    else if !specs.forall(_.hasEdgeRendering) then
      Left("TesseractEdgeSceneBuilder requires edge rendering parameters on all specs")
    else
      // Check compatibility - all specs must have same 4D projection params
      val firstSpec = specs.head
      specs.find(!isCompatible(_, firstSpec)) match
        case Some(incompatible) =>
          Left(s"Incompatible 4D projection parameters between tesseracts. " +
            "All tesseracts must have matching projection parameters for shared mesh rendering.")
        case None =>
          // Calculate total instances: faces (1 mesh instance per spec) + edges (32 cylinders per spec)
          val totalInstances = calculateInstanceCount(specs)
          if totalInstances > maxInstances then
            Left(s"Too many instances: $totalInstances exceeds max instances limit of $maxInstances. " +
              "Each tesseract with edges creates 33 instances (1 mesh + 32 cylinders).")
          else
            Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer): Try[Unit] = Try:
    logger.info(s"Building tesseract scene with edge rendering: ${specs.length} tesseracts")

    // Create shared base geometry for faces
    val firstSpec = specs.head
    val mesh = MeshFactory.create(firstSpec)
    renderer.setTriangleMesh(mesh)

    // Load textures
    val textureIndices = TextureManager.loadTextures(specs, renderer, textureDir)

    // Add instances for each tesseract
    specs.foreach { spec =>
      // Add face mesh instance
      val faceMaterial = MaterialExtractor.extract(spec)
      val position = Vector[3](spec.x, spec.y, spec.z)
      val textureIndex = spec.texture.flatMap(textureIndices.get).getOrElse(-1)

      renderer.addTriangleMeshInstance(position, faceMaterial, textureIndex) match
        case Some(id) =>
          logger.debug(s"Added tesseract face mesh instance $id at ($position)")
        case None =>
          logger.error(s"Failed to add tesseract face mesh instance at ($position)")

      // Add edge cylinder instances
      addEdgeCylinders(spec, renderer)
    }

  /**
   * Add cylinder instances for all edges of a tesseract.
   *
   * A tesseract has 32 edges. Each edge is projected from 4D to 3D using
   * the same rotation and projection as the faces.
   */
  private def addEdgeCylinders(spec: ObjectSpec, renderer: OptiXRenderer): Unit =
    val proj4D = spec.projection4D.getOrElse(Projection4DSpec.default)
    val edgeRadius = spec.edgeRadius.getOrElse(0.02f)
    val edgeMaterial = spec.edgeMaterial.getOrElse(defaultEdgeMaterial)

    // Create tesseract with spec's size
    val tesseract = Tesseract(size = spec.size)

    // Create rotation and projection (same as TesseractMesh)
    val rotation: Rotation =
      if proj4D.rotXW == 0f && proj4D.rotYW == 0f && proj4D.rotZW == 0f then
        Rotation.identity
      else
        Rotation(proj4D.rotXW, proj4D.rotYW, proj4D.rotZW, Vector[4](0f, 0f, 0f, 0f))

    val projection = Projection(proj4D.eyeW, proj4D.screenW)

    // Position offset for this tesseract instance
    val offset = Vector3(spec.x, spec.y, spec.z)

    // Project each edge and create cylinder
    val edgeCount = tesseract.edges.count { case (v0_4d, v1_4d) =>
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

    logger.debug(s"Added $edgeCount edge cylinders for tesseract at (${spec.x}, ${spec.y}, ${spec.z})")

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    // Both must be hypercubes
    if !ObjectType.isHypercube(spec1.objectType) || !ObjectType.isHypercube(spec2.objectType) then
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
    // Each tesseract: 1 face mesh instance + 32 edge cylinder instances = 33 instances
    specs.length.toLong * 33L
