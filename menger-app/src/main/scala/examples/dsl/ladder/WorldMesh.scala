package examples.dsl.ladder

import menger.common.ProfilingConfig
import menger.dsl.SceneObject
import menger.engines.scene.MeshFactory
import menger.engines.scene.TriangleMeshSceneBuilder
import menger.spec.ObjectSpec

/** An object as the renderer places it: its mesh from `MeshFactory`, moved by the instance
  * transform. The one place the ladder's tests read world coordinates from. */
object WorldMesh:

  type Point = (Float, Float, Float)

  val vertexTolerance = 1e-3f
  private val degenerateArea = 1e-9f

  /** The corners of the object's non-degenerate triangles, in world coordinates. */
  def corners(spec: ObjectSpec): Iterator[Point] =
    given ProfilingConfig = ProfilingConfig.disabled
    val mesh = MeshFactory.create(spec)
    val m = TriangleMeshSceneBuilder.instanceTransform(spec)
    def at(i: Int): Point =
      val base = i * mesh.vertexStride
      val (x, y, z) = (mesh.vertices(base), mesh.vertices(base + 1), mesh.vertices(base + 2))
      (m(0) * x + m(1) * y + m(2) * z + m(3), m(4) * x + m(5) * y + m(6) * z + m(7),
        m(8) * x + m(9) * y + m(10) * z + m(11))
    def area(a: Point, b: Point, c: Point): Float =
      val (ux, uy, uz) = (b._1 - a._1, b._2 - a._2, b._3 - a._3)
      val (vx, vy, vz) = (c._1 - a._1, c._2 - a._2, c._3 - a._3)
      val (cx, cy, cz) = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
      0.5f * math.sqrt(cx * cx + cy * cy + cz * cz).toFloat
    mesh.indices.grouped(3).map(_.map(at)).filter(t => area(t(0), t(1), t(2)) > degenerateArea)
      .flatten

  /** The distinct corners of the object's non-degenerate triangles, in world coordinates. */
  def vertices(obj: SceneObject): Seq[Point] =
    corners(obj.toObjectSpec).foldLeft(List.empty[Point]): (kept, p) =>
      if kept.exists(q => near(p, q)) then kept else p :: kept

  def near(p: Point, q: Point): Boolean =
    math.abs(p._1 - q._1) < vertexTolerance && math.abs(p._2 - q._2) < vertexTolerance &&
      math.abs(p._3 - q._3) < vertexTolerance
