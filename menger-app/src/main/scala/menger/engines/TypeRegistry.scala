package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType
import menger.common.ProfilingConfig
import menger.engines.scene.*

/** Single source of truth for object-type → builder/classification mapping.
  *
  * Adding a new geometry type touches only:
  * 1. `ObjectType.VALID_TYPES` (in menger-common)
  * 2. One new `SceneBuilder` subclass
  * 3. This registry — one entry in the `entries` map
  *
  * `GeometryRegistry` and `RenderModeSelector` both consume this table,
  * eliminating the duplicated hand-maintained if/else chains (T1, Sprint 32).
  */
object TypeRegistry:

  /** A descriptor for one object type: how to build it and how to classify it. */
  case class Entry(
    /** Human-readable type name (matches ObjectType.VALID_TYPES entry). */
    name: String,
    /** Produces a SceneBuilder for a homogeneous group of this type. */
    builderFactory: (String, ProfilingConfig) => SceneBuilder,
    /** Classifies a list of specs of this type into a SceneType. */
    sceneTypeFactory: List[ObjectSpec] => SceneType,
    /** True if this is a 4D projected type (menger4d, sierpinski4d, etc.). */
    isProjected4D: Boolean = false,
    /** True if this type supports edge rendering (tesseract). */
    supportsEdgeRendering: Boolean = false,
    /** True if this is an analytical primitive (sphere, cylinder, cone, curve). */
    isAnalytical: Boolean = false
  )

  private def sphereEntry = Entry("sphere",
    (td, _) => SphereSceneBuilder(td),
    specs => SceneType.SimpleMixed(specs, "sphere"),
    isAnalytical = true)
  private def curveEntry = Entry("curve",
    (td, _) => CurveSceneBuilder(td),
    specs => SceneType.Curves(specs),
    isAnalytical = true)
  private def cubeSpongeEntry = Entry("cube-sponge",
    (td, _) => CubeSpongeSceneBuilder(td),
    specs => SceneType.SimpleMixed(specs, "cube-sponge"))
  private def coneEntry = Entry("cone",
    (td, _) => ConeSceneBuilder(td),
    specs => SceneType.SimpleMixed(specs, "cone"),
    isAnalytical = true)
  private def planeEntry = Entry("plane",
    (td, _) => PlaneSceneBuilder(td),
    specs => SceneType.SimpleMixed(specs, "plane"))
  private def menger4DEntry = Entry("menger4d",
    (td, _) => Menger4DSceneBuilder(td),
    specs => SceneType.Menger4D(specs),
    isProjected4D = true)
  private def sierpinski4DEntry = Entry("sierpinski4d",
    (td, _) => Sierpinski4DSceneBuilder(td),
    specs => SceneType.Sierpinski4D(specs),
    isProjected4D = true)
  private def hexadecachoron4DEntry = Entry("hexadecachoron4d",
    (td, _) => Hexadecachoron4DSceneBuilder(td),
    specs => SceneType.Hexadecachoron4D(specs),
    isProjected4D = true)
  private def triangleMeshEntry = Entry("__triangle_mesh__",
    (td, pc) => TriangleMeshSceneBuilder(td)(using pc),
    specs => SceneType.TriangleMeshes(specs))

  val allEntries: List[Entry] = List(
    sphereEntry, curveEntry, cubeSpongeEntry, coneEntry, planeEntry,
    menger4DEntry, sierpinski4DEntry, hexadecachoron4DEntry, triangleMeshEntry
  )

  /** All registered type entries, indexed by normalized name. */
  val entries: Map[String, Entry] =
    allEntries.map(e => e.name -> e).toMap

  /** Returns the entry for a given normalized type name, or None. */
  def forType(typeName: String): Option[Entry] =
    entries.get(typeName).orElse(
      if ObjectType.isTriangleMesh(typeName) then entries.get("__triangle_mesh__")
      else None
    )

  /** All built-in (non-triangle-mesh) type names. */
  val builtInTypeNames: Set[String] =
    entries.keySet - "__triangle_mesh__"

  /** Triangle-mesh type names (the catch-all). */
  def isTriangleMeshType(name: String): Boolean =
    !builtInTypeNames.contains(name) && ObjectType.isTriangleMesh(name)
