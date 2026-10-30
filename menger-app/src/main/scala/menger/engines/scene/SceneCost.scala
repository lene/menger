package menger.engines.scene

import menger.ObjectSpec
import menger.common.Const
import menger.dsl.ResourceLimits
import menger.geometry.LevelMix
import menger.objects.SpongeByVolume
import menger.objects.higher_d.TesseractSponge2Mesh
import menger.objects.higher_d.TesseractSpongeMesh
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
    * the instance table) count zero; a pre-tessellated mesh counts its own triangles. */
  def triangles(spec: ObjectSpec): Long =
    val byLevel = for
      estimate <- levelTriangles.get(ObjectType.normalize(spec.objectType))
      level    <- spec.level
    yield
      val lower = level.floor.toInt
      if level == level.floor then estimate(lower) else estimate(lower) + estimate(lower + 1)
    byLevel.getOrElse(spec.meshData.map(_.indices.length.toLong / 3).getOrElse(0L))

  /** Whole renders per displayed frame: one per level-mix world, each repeated per accumulation
    * frame. */
  def rendersPerFrame(specs: List[ObjectSpec], accumulation: Int): Long =
    val mixed = specs.count(spec => LevelMixing.appliesTo(spec.level, MaterialExtractor.extract(spec)))
    val worlds = if mixed == 0 then 1 else LevelMix.worlds(Vector.fill(mixed)(0.5f)).size
    worlds.toLong * math.max(1, accumulation).toLong

  def edgeTubeSpongeLevel(specs: List[ObjectSpec]): Int =
    specs.filter(MeshEdges3D.appliesTo).flatMap(_.level).map(_.ceil.toInt).maxOption.getOrElse(0)

  def fittingLevel(specs: List[ObjectSpec]): Option[FittingLevel] =
    val total = specs.map(triangles).sum
    val levelled = for
      spec     <- specs
      perLevel <- levelTriangles.get(ObjectType.normalize(spec.objectType))
      level    <- spec.level
    yield (spec, perLevel, level)
    if total <= ResourceLimits.sceneMaxTriangles then None
    else
      for
        (spec, perLevel, level) <- levelled.maxByOption((spec, _, _) => triangles(spec))
        others = total - triangles(spec)
        fit <- (level.floor.toInt to 0 by -1).find(l =>
          others + perLevel(l) <= ResourceLimits.sceneMaxTriangles)
      yield FittingLevel(ObjectType.normalize(spec.objectType), fit, others + perLevel(fit))

  def estimate(specs: List[ObjectSpec], accumulation: Int, caustics: CausticsLoad): Estimate =
    val renders = rendersPerFrame(specs, accumulation)
    Estimate(specs.map(triangles).sum, renders, caustics.photonsPerRender * renders,
      edgeTubeSpongeLevel(specs), fittingLevel(specs))

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
    check(Estimate(specs.map(triangles).sum, 1L, 0L, fittingLevel = fittingLevel(specs)))
