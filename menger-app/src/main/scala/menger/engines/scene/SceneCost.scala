package menger.engines.scene

import menger.common.Const
import menger.dsl.ResourceLimits
import menger.geometry.LevelMix
import menger.objects.SpongeByVolume
import menger.objects.higher_d.NCube
import menger.objects.higher_d.NdSponge
import menger.objects.higher_d.NdSpongeVariant
import menger.objects.higher_d.TesseractSponge2Mesh
import menger.objects.higher_d.TesseractSpongeMesh
import menger.spec.ObjectSpec
import menger.spec.ObjectType

/** What a scene costs as a whole, against `ResourceLimits`' scene-wide budget (review
  * 2026-10-14, menger#80). Each DSL field has its own cap, but the caps multiply: two
  * level-5 volume sponges each pass, together they run the renderer out of memory; fractional
  * glass objects multiply the renders per frame, and accumulation and caustics multiply each
  * render. Pure arithmetic over the object specs -- nothing is built -- so the validator can
  * reject an over-budget scene before building its meshes, and every engine can refuse one
  * before it allocates them.
  */
object SceneCost:

  /** The most expensive levelled object of a scene over the triangle budget, and the highest
    * whole level at which the scene would fit (usability review 2026-10, session 6, msa#48: a
    * rejection that names only the totals leaves the user to guess a level). */
  final case class FittingLevel(objectType: String, level: Int, sceneTriangles: Long)

  /** `edgeTubeSpongeLevel`: the highest whole level whose edge tubes a 3D sponge draws. */
  final case class Estimate(triangles: Long, rendersPerFrame: Long, photonsPerFrame: Long,
    edgeTubeSpongeLevel: Int = 0, fittingLevel: Option[FittingLevel] = None)

  /** Caustics as far as cost is concerned: photons traced per render, or none. */
  final case class CausticsLoad(photonsPerIteration: Long, iterations: Long):
    def photonsPerRender: Long = photonsPerIteration * iterations

  object CausticsLoad:
    val None: CausticsLoad = CausticsLoad(0L, 0L)

  /** Surface-subdivision sponge: 12^n cubes' worth of faces (AD-1), the estimate the
    * interactive engine's slowness warning has always used. */
  def surfaceSpongeTriangles(level: Int): Long =
    math.pow(Const.Engine.trianglesPerCube, level).toLong * 6 * 2

  private val levelTriangles: Map[String, Int => Long] = Map(
    "sponge-volume"            -> SpongeByVolume.triangleCount,
    "sponge-surface"           -> surfaceSpongeTriangles,
    "tesseract-sponge"         -> TesseractSpongeMesh.estimatedTriangles,
    "tesseract-sponge-volume"  -> TesseractSpongeMesh.estimatedTriangles,
    "tesseract-sponge-2"       -> TesseractSponge2Mesh.estimatedTriangles,
    "tesseract-sponge-surface" -> TesseractSponge2Mesh.estimatedTriangles,
  )

  /** Triangles one object holds. A fractional level keeps both whole levels (the level mix
    * draws both; the cap fade builds level n+1 plus level n's caps), so both are counted. Objects
    * without a CPU mesh (analytical primitives, GPU-procedural 4D, instanced sponges bounded by
    * the instance table) count zero; a pre-tessellated mesh counts its own triangles. An
    * n-sponge volume counts exactly, an n-sponge surface by its cheap bound while that is within
    * the budget, else exactly, both without building a mesh (see [[ndSpongeTriangles]]); a scene
    * of them is judged by [[costs]]. */
  def triangles(spec: ObjectSpec): Long = triangles(spec, exact = false)

  private def triangles(spec: ObjectSpec, exact: Boolean): Long =
    if ObjectType.isNCube(spec.objectType) then nCubeTriangles(spec) else meshTriangles(spec, exact)

  /** An n-cube's square faces, two triangles each (those clipped at an eye count too). */
  private def nCubeTriangles(spec: ObjectSpec): Long =
    spec.dimension.fold(0L)(n => ResourceLimits.TrianglesPerFace * NCube.faceCount(n))

  /** The triangles of an n-sponge at a whole level; no mesh is built to find out. The volume's
    * faces are always counted exactly, by a recurrence that costs nothing. The surface's cost is
    * its cheap upper bound while that is within the budget (or, with `exact`, never), else its
    * faces counted on face keys alone. A level past `ResourceLimits`' headroom, or one a build
    * could not allocate, is not counted: it costs its bound, which is over the budget. */
  private def ndSpongeTriangles(spec: ObjectSpec, level: Int, exact: Boolean): Long =
    val variant = NdSpongeScene.variantOf(spec)
    val k = NdSpongeScene.maxCentreCoordinatesOf(spec)
    val n = NCubeScene.dimensionOf(spec)
    val bound = NdSponge.upperBoundFaces(variant, n, level, k)
    val boundTriangles = saturatingTimes(bound, ResourceLimits.TrianglesPerFace)
    val countable = ResourceLimits.ndSpongeBoundAdmits(bound) &&
      NdSponge.isBuildable(variant, n, level, k)
    def counted =
      saturatingTimes(NdSponge.faceCount(variant, n, level, k), ResourceLimits.TrianglesPerFace)
    variant match
      case _ if !countable => math.min(Long.MaxValue / RefusedSummands, boundTriangles)
      case NdSpongeVariant.Volume => counted
      case NdSpongeVariant.Surface =>
        if !exact && boundTriangles <= ResourceLimits.sceneMaxTriangles then boundTriangles
        else counted

  // a refused sponge costs about this much, kept small enough to add a few of them up
  private val RefusedSummands = 1024L

  private def saturatingTimes(a: Long, b: Long): Long =
    if a > Long.MaxValue / b then Long.MaxValue else a * b

  /** The triangles per whole level of a levelled mesh type, None for the other types. */
  private def trianglesAtLevel(spec: ObjectSpec, exact: Boolean): Option[Int => Long] =
    if ObjectType.isNdSponge(spec.objectType) then Some(ndSpongeTriangles(spec, _, exact))
    else levelTriangles.get(ObjectType.normalize(spec.objectType))

  private def meshTriangles(spec: ObjectSpec, exact: Boolean): Long =
    val byLevel = for
      estimate <- trianglesAtLevel(spec, exact)
      level    <- spec.level
    yield
      val lower = level.floor.toInt
      if level == level.floor then estimate(lower) else estimate(lower) + estimate(lower + 1)
    byLevel.getOrElse(spec.meshData.map(_.indices.length.toLong / 3).getOrElse(0L))

  /** The triangles of each object of a scene, and whether n-sponges were counted exactly. Their
    * surface bound is loose (about 1.4 times the count for the 6D surface at level 3), so when
    * the bounds add up to more than the budget they are counted exactly (within the bound's
    * headroom) before the scene is judged: sponges that fit must not be refused for the sum of
    * their bounds. */
  private def costs(specs: List[ObjectSpec]): (List[Long], Boolean) =
    val byBound = specs.map(triangles(_, exact = false))
    val hasNdSponge = specs.exists(spec => ObjectType.isNdSponge(spec.objectType))
    if byBound.sum <= ResourceLimits.sceneMaxTriangles || !hasNdSponge then (byBound, false)
    else (specs.map(triangles(_, exact = true)), true)

  /** Whole renders per displayed frame: one per level-mix world, each repeated per accumulation
    * frame. */
  def rendersPerFrame(specs: List[ObjectSpec], accumulation: Int): Long =
    val mixed = specs.count(spec => LevelMixing.appliesTo(spec.level, MaterialExtractor.extract(spec)))
    val worlds = if mixed == 0 then 1 else LevelMix.worlds(Vector.fill(mixed)(0.5f)).size
    worlds.toLong * math.max(1, accumulation).toLong

  def edgeTubeSpongeLevel(specs: List[ObjectSpec]): Int =
    specs.filter(MeshEdges3D.appliesTo).flatMap(_.level).map(_.ceil.toInt).maxOption.getOrElse(0)

  def fittingLevel(specs: List[ObjectSpec]): Option[FittingLevel] = fittingLevel(specs, costs(specs))

  private def fittingLevel(
    specs: List[ObjectSpec], scene: (List[Long], Boolean)
  ): Option[FittingLevel] =
    val (perObject, exact) = scene
    val total = perObject.sum
    val levelled = for
      (spec, cost) <- specs.zip(perObject)
      perLevel     <- trianglesAtLevel(spec, exact)
      level        <- spec.level
    yield (spec, cost, perLevel, level)
    if total <= ResourceLimits.sceneMaxTriangles then None
    else
      for
        (spec, cost, perLevel, level) <- levelled.maxByOption((_, cost, _, _) => cost)
        others = total - cost
        fit <- (level.floor.toInt to 0 by -1).find(l =>
          others + perLevel(l) <= ResourceLimits.sceneMaxTriangles)
      yield FittingLevel(ObjectType.normalize(spec.objectType), fit, others + perLevel(fit))

  def estimate(specs: List[ObjectSpec], accumulation: Int, caustics: CausticsLoad): Estimate =
    val renders = rendersPerFrame(specs, accumulation)
    val scene = costs(specs)
    Estimate(scene._1.sum, renders, caustics.photonsPerRender * renders,
      edgeTubeSpongeLevel(specs), fittingLevel(specs, scene))

  /** `Left` names every budget the scene exceeds and what to lower; `Right` when it fits. */
  def check(estimate: Estimate): Either[String, Unit] =
    val problems = List(
      Option.when(estimate.triangles > ResourceLimits.sceneMaxTriangles)(
        s"${estimate.triangles} triangles in total, budget ${ResourceLimits.sceneMaxTriangles} " +
          "(lower sponge levels, or use fewer high-level objects)" +
          estimate.fittingLevel.fold("")(fit =>
            s"; the ${fit.objectType} fits at level ${fit.level} " +
              s"(${fit.sceneTriangles} triangles in total)")),
      Option.when(estimate.rendersPerFrame > ResourceLimits.sceneMaxRendersPerFrame)(
        s"${estimate.rendersPerFrame} renders per frame, budget " +
          s"${ResourceLimits.sceneMaxRendersPerFrame} (each fractional-level glass object doubles " +
          "the renders up to 16, accumulation multiplies them; lower accumulation, or use whole " +
          "levels)"),
      Option.when(estimate.photonsPerFrame > ResourceLimits.sceneMaxPhotonsPerFrame)(
        s"${estimate.photonsPerFrame} caustics photons per frame, budget " +
          s"${ResourceLimits.sceneMaxPhotonsPerFrame} (lower photonsPerIteration, iterations or " +
          "accumulation)"),
      Option.when(estimate.edgeTubeSpongeLevel > ResourceLimits.edgeTubeSpongeMaxLevel)(
        s"edge tubes on a sponge of level ${estimate.edgeTubeSpongeLevel}, at most level " +
          s"${ResourceLimits.edgeTubeSpongeMaxLevel} (one tube per edge: level 4 has 427,728; " +
          "lower the level to 3 or less, or drop edgeRadius/edgeMaterial)")
    ).flatten
    if problems.isEmpty then Right(()) else Left(s"scene over budget: ${problems.mkString("; ")}")

  /** Triangles only: what an engine can check from the specs alone, before allocating meshes. */
  def checkTriangles(specs: List[ObjectSpec]): Either[String, Unit] =
    val scene = costs(specs)
    check(Estimate(scene._1.sum, 1L, 0L, fittingLevel = fittingLevel(specs, scene)))
