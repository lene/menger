package menger.engines.scene

import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.common.Material
import menger.common.ObjectType
import menger.common.ProfilingConfig
import menger.common.TriangleMeshData
import menger.common.Vector

/** Edge tubes of a 3D triangle-mesh object: a Cube, a platonic solid or a mesh Sponge
  * (usability review 2026-10, session 5, F124, menger#132: only the 4D types had edges).
  * An edge is where the mesh's
  * triangles meet at an angle: a quad's diagonal and the seam between two coplanar faces of
  * neighbouring cubes are not edges. One cylinder per edge, as for the 4D types; a recursive-IAS
  * sponge has no flat mesh to take them from. */
object MeshEdges3D:

  def appliesTo(spec: ObjectSpec): Boolean =
    spec.hasEdgeRendering && ObjectSpec.Edge3DTypes.contains(ObjectType.normalize(spec.objectType))

  // Vertex positions are compared on this grid: a mesh repeats each corner once per face.
  private val Quantum = 1e-4f

  private type Key = (Long, Long, Long)

  private def key(x: Float, y: Float, z: Float): Key =
    (math.round(x / Quantum).toLong, math.round(y / Quantum).toLong,
      math.round(z / Quantum).toLong)

  /** The mesh's edges, in its own coordinates. */
  def of(mesh: TriangleMeshData): IndexedSeq[(Vector[3], Vector[3])] =
    val v = mesh.vertices
    val stride = mesh.vertexStride
    def position(i: Int) = Vector[3](v(i * stride), v(i * stride + 1), v(i * stride + 2))
    def keyOf(p: Vector[3]) = key(p(0), p(1), p(2))
    // Every triangle side with its triangle's (quantized) normal.
    val sides = mesh.indices.grouped(3).collect { case Array(a, b, c) =>
      val (pa, pb, pc) = (position(a), position(b), position(c))
      val normal = unitNormal(pa, pb, pc)
      Seq((pa, pb, normal), (pb, pc, normal), (pc, pa, normal))
    }.flatten.toVector
    sides.groupBy { (p, q, _) =>
      val (kp, kq) = (keyOf(p), keyOf(q))
      if Ordering[Key].lteq(kp, kq) then (kp, kq) else (kq, kp)
    }.values.collect {
      case group if group.size == 1 || group.map(_._3).distinct.size > 1 =>
        (group.head._1, group.head._2)
    }.toIndexedSeq

  private def unitNormal(a: Vector[3], b: Vector[3], c: Vector[3]): Key =
    val (ux, uy, uz) = (b(0) - a(0), b(1) - a(1), b(2) - a(2))
    val (vx, vy, vz) = (c(0) - a(0), c(1) - a(1), c(2) - a(2))
    val (nx, ny, nz) = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    val length = math.sqrt(nx * nx + ny * ny + nz * nz).toFloat
    if length == 0f then (0L, 0L, 0L)
    else key(nx / length, ny / length, nz / length)

  /** `spec`'s edges in world coordinates: the instance transform the faces get. */
  def worldEdges(spec: ObjectSpec)(using ProfilingConfig): IndexedSeq[(Vector[3], Vector[3])] =
    val m = TriangleMeshSceneBuilder.instanceTransform(spec)
    def place(p: Vector[3]) = Vector[3](
      m(0) * p(0) + m(1) * p(1) + m(2) * p(2) + m(3),
      m(4) * p(0) + m(5) * p(1) + m(6) * p(2) + m(7),
      m(8) * p(0) + m(9) * p(1) + m(10) * p(2) + m(11)
    )
    of(MeshFactory.create(spec)).map((p, q) => (place(p), place(q)))

  /** Each level whose edges are drawn for `spec` with the share it keeps: a fractional level
    * cross-fades its two levels' edges, as the 4D types do. */
  private def levels(spec: ObjectSpec): List[(ObjectSpec, Float)] =
    if spec.level.isEmpty then List((spec, 1f)) else TesseractEdgeSceneBuilder.edgeLevels(spec)

  /** How many cylinders `spec`'s edges need. */
  def count(spec: ObjectSpec)(using ProfilingConfig): Long =
    if !appliesTo(spec) then 0L
    else levels(spec).map((levelSpec, _) => of(MeshFactory.create(levelSpec)).size.toLong).sum

  /** Adds one cylinder per edge of `spec`; returns the renderer's raw instance ids. */
  def add(spec: ObjectSpec, renderer: OptiXRenderer)(using ProfilingConfig): Seq[Int] =
    val radius = spec.edgeRadius.getOrElse(TesseractEdgeSceneBuilder.DefaultEdgeRadius)
    val material = spec.edgeMaterial.getOrElse(Material.Film)
    levels(spec).flatMap { (levelSpec, keep) =>
      val faded = TesseractEdgeSceneBuilder.fadedEdgeMaterial(material, keep)
      worldEdges(levelSpec).map((p, q) => renderer.addCylinderInstance(p, q, radius, faded))
    }
