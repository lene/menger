package menger.engines

import menger.ObjectSpec
import menger.ProfilingConfig
import menger.common.ObjectType
import menger.engines.scene.CubeSpongeSceneBuilder
import menger.engines.scene.SceneBuilder
import menger.engines.scene.SphereSceneBuilder
import menger.engines.scene.TesseractEdgeSceneBuilder
import menger.engines.scene.TriangleMeshSceneBuilder

/** Scene classification and builder selection logic shared by OptiXEngine and AnimatedOptiXEngine. */
object SceneClassifier:

  def classify(specs: List[ObjectSpec]): SceneType =
    require(specs.nonEmpty, "classify requires at least one ObjectSpec")
    val types = specs.map(_.objectType.toLowerCase).toSet

    if types.contains("cube-sponge") then
      SceneType.CubeSponges(specs)
    else if types.forall(_ == "sphere") then
      SceneType.Spheres(specs)
    else if types.forall(isTriangleMeshType) then
      SceneType.TriangleMeshes(specs)
    else
      // Mixed scene - spheres + triangle meshes
      val hasSpheres  = types.contains("sphere")
      val meshTypes   = types.filter(isTriangleMeshType)

      if hasSpheres && meshTypes.size == 1 then
        // Simple mixed: spheres + one mesh type (SUPPORTED)
        SceneType.SimpleMixed(specs, meshTypes.head)
      else if hasSpheres && meshTypes.size > 1 then
        // Check if all mesh types are 4D projected and compatible
        val all4DProjected = meshTypes.forall(ObjectType.isProjected4D)
        if all4DProjected then
          // All 4D objects can share GAS - treat as SimpleMixed with first type
          SceneType.SimpleMixed(specs, meshTypes.head)
        else
          // Complex mixed: spheres + multiple incompatible mesh types (NOT SUPPORTED)
          SceneType.ComplexMixed(specs)
      else
        // Other mixed scenarios
        SceneType.ComplexMixed(specs)

  def isTriangleMeshType(objectType: String): Boolean =
    val t = objectType.toLowerCase
    t == "cube" ||
    ObjectType.isSponge(t) ||
    ObjectType.isProjected4D(t)

  def selectSceneBuilder(
    sceneType: SceneType,
    textureDir: Option[String]
  )(using ProfilingConfig): Option[SceneBuilder] =
    val dir = textureDir.getOrElse(".")
    sceneType match
      case SceneType.Spheres(_)       => Some(SphereSceneBuilder())
      case SceneType.CubeSponges(_)   => Some(CubeSpongeSceneBuilder())
      case SceneType.TriangleMeshes(specs) =>
        val all4DProjected   = specs.forall(s => ObjectType.isProjected4D(s.objectType))
        val hasEdgeRendering = specs.exists(_.hasEdgeRendering)
        if all4DProjected && hasEdgeRendering then
          Some(TesseractEdgeSceneBuilder(dir))
        else
          Some(TriangleMeshSceneBuilder(dir))
      case SceneType.SimpleMixed(_, _) => None  // Handled specially in createMultiObjectScene
      case SceneType.ComplexMixed(_)   => None

enum SceneType:
  case CubeSponges(specs: List[ObjectSpec])
  case Spheres(specs: List[ObjectSpec])
  case TriangleMeshes(specs: List[ObjectSpec])
  case SimpleMixed(specs: List[ObjectSpec], meshType: String)
  case ComplexMixed(specs: List[ObjectSpec])
