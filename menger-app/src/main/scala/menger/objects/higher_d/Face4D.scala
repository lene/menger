package menger.objects.higher_d

import scala.annotation.targetName

import menger.common.Const
import menger.common.Vector

case class Face4D[V <: Int & Singleton](vertices: IndexedSeq[Vector[4]])(using v: ValueOf[V]):

  val vertsPerFace: Int = v.value
  require(vertices.size == vertsPerFace,
    s"Face4D[$vertsPerFace] requires $vertsPerFace vertices, got ${vertices.size}")

  def apply(i: Int): Vector[4] = vertices(i)
  def asSeq: Seq[Vector[4]] = vertices.toSeq

  def area: Float =
    if vertsPerFace < 3 then 0f
    else
      val u = vertices(1) - vertices(0)
      val v = vertices(2) - vertices(0)
      val u2 = u * u
      val v2 = v * v
      val uv = u * v
      val area2 = u2 * v2 - uv * uv
      if area2 < 0f then 0f else math.sqrt(area2).toFloat

  def map(f: Vector[4] => Vector[4]): Face4D[V] =
    Face4D[V](vertices.map(f))

  @targetName("plus")
  def +(delta: Vector[4]): Face4D[V] = map(_ + delta)

  @targetName("div")
  def /(scale: Float): Face4D[V] = map(_ / scale)

  @targetName("equals")
  def ==(that: Face4D[V]): Boolean = asSeq == that.asSeq

  override def toString: String = asSeq.map(_.toString).mkString("(", ", ", ")")

object Face4D:

  given v3: ValueOf[3] = ValueOf(3)
  given v4: ValueOf[4] = ValueOf(4)
  given v5: ValueOf[5] = ValueOf(5)

  val dimension: Int = 4
  val numVertices: Int = 4

  def apply(a: Vector[4], b: Vector[4], c: Vector[4], d: Vector[4]): Face4D[4] =
    Face4D[4](IndexedSeq(a, b, c, d))

  def apply(vertices: Seq[Vector[4]]): Face4D[4] =
    require(vertices.length == 4, s"Need 4 vertices, have ${vertices.length}")
    val squaredEdgeLengths = (vertices :+ vertices.head).sliding(2).map(
      edge => edge.head.dst2(edge.last)
    ).toSet
    require(
      squaredEdgeLengths.max - squaredEdgeLengths.min <= Const.epsilon,
      s"Vertices must all be same length, are $squaredEdgeLengths"
    )
    Face4D[4](vertices.toIndexedSeq)

  def fromSeq(vertices: Seq[Vector[4]]): Face4D[4] =
    require(vertices.length == 4, s"Need 4 vertices, have ${vertices.length}")
    val squaredEdgeLengths = (vertices :+ vertices.head).sliding(2).map(
      edge => edge.head.dst2(edge.last)
    ).toSet
    require(
      squaredEdgeLengths.max - squaredEdgeLengths.min <= Const.epsilon,
      s"Vertices must all be same length, are $squaredEdgeLengths"
    )
    Face4D[4](vertices.toIndexedSeq)

  export Face4DHelpers.normalDirections
  export Face4DHelpers.normalSigns
  export Face4DHelpers.vectorIndices
  export Face4DHelpers.allIndicesWhere
  export Face4DHelpers.remainingCorners

// Quad-specific backward-compatible extensions
extension (f: Face4D[4])
  def a: Vector[4] = f(0)
  def b: Vector[4] = f(1)
  def c: Vector[4] = f(2)
  def d: Vector[4] = f(3)
  def asTuple: (Vector[4], Vector[4], Vector[4], Vector[4]) = (f(0), f(1), f(2), f(3))
  def edges: Seq[Edge] = Seq(Edge(f(0), f(1)), Edge(f(1), f(2)), Edge(f(2), f(3)), Edge(f(3), f(0)))
  def plane: Plane = Plane(f.asSeq)

  def normals: Seq[Vector[4]] =
    val e = Seq(f(1) - f(0), f(2) - f(1), f(3) - f(2), f(0) - f(3))
    require(
      e.map(v => v.count(_.abs > Const.epsilon) == 1).forall(_ == true),
      s"Edges must be parallel to the axes, are ${e.map(_.toString)}"
    )
    require(
      e.sliding(2).forall({ case Seq(a, b) => a * b < Const.epsilon }),
      s"Edges must be orthogonal, are ${e.map(_.toString)}"
    )
    val dirs = Face4DHelpers.normalDirections(e, 4)
    val signs = Face4DHelpers.normalSigns(e.take(2))
    dirs.zip(signs).map { case (vec, sign) => vec * sign }

  def extrude(): Seq[Face4D[4]] = edges.flatMap { edge => extrude(edge) }

  def extrude(edge: Edge): Seq[Face4D[4]] =
    val allCorners = f.asSeq
    val corners = edge.asSeq
    val oppositeCorners = Face4DHelpers.remainingCorners(allCorners, corners)
    require(oppositeCorners.size == 2,
      s"${corners.map(_.toString)} not in corner set")
    val distance = oppositeCorners.head - corners.last
    require(
      distance.dst(oppositeCorners.last - corners.head) < Const.epsilon,
      "Corners must be opposite")
    val parentCentre = (f(0) + f(1) + f(2) + f(3)) / 4f
    val edgeMid = (edge.v0 + edge.v1) / 2f
    val qOffset = edgeMid - parentCentre
    val qAxisOpt = (0 until 4).find(i => qOffset(i).abs > Const.epsilon)
    require(qAxisOpt.isDefined, s"edge midpoint coincides with parent centre in $f")
    val qAxis = qAxisOpt.get
    val qNormal = Vector.unit[4](qAxis) * -qOffset(qAxis).sign
    val parentNormals = f.normals
    parentNormals.zipWithIndex.map { case (extrusionNormal, i) =>
      val siblingNormal = parentNormals(1 - i)
      val expected = Set(qNormal, siblingNormal)
      val newOpposites = corners.map(_ + extrusionNormal * distance.len)
      val base = Seq(edge.v0, edge.v1, newOpposites.last, newOpposites.head)
      val chosen = (0 until 4).iterator.map { k =>
        val r = (0 until 4).map(j => base((j + k) % 4))
        Face4D[4](r.toIndexedSeq)
      }.find(_.normals.toSet == expected)
      require(chosen.isDefined,
        s"No cyclic ordering of $base produces expected normals $expected")
      chosen.get
    }

object Face4DHelpers:
  def normalDirections(edgeVectors: Seq[Vector[4]], numVertices: Int): Seq[Vector[4]] =
    val edgeDirectionIndices = edgeVectors.toSet.flatMap(vectorIndices)
    val normalIndices = (0 until numVertices).toSet.diff(edgeDirectionIndices)
    normalIndices.map(Vector.unit[4]).toSeq

  def normalSigns(edgeVectors: Seq[Vector[4]]): Seq[Float] =
    val sum = edgeVectors.reduce(_ + _)
    sum.filter(_.abs > Const.epsilon).map(_.sign)

  def vectorIndices(v: Vector[4]): Seq[Int] =
    allIndicesWhere(v.toIndexedSeq, s => s.abs > Const.epsilon)

  def allIndicesWhere[A](s: Seq[A], pred: A => Boolean): Seq[Int] =
    s.zipWithIndex.filter { case (elem, _) => pred(elem) }.map { case (_, index) => index }

  def remainingCorners(allCorners: Seq[Vector[4]], cornersToRemove: Seq[Vector[4]]): Seq[Vector[4]] =
    require(allCorners.size == 4, s"Need 4 corners, have ${allCorners.size}")
    require(cornersToRemove.size == 2, s"Need 2 corners to remove, have ${cornersToRemove.size}")
    val all = allCorners.toVector
    val firstToRemove = all.indexWhere(_.dst(cornersToRemove.head) < Const.epsilon)
    val secondToRemove = all.indexWhere(_.dst(cornersToRemove.last) < Const.epsilon)
    require(
      secondToRemove == firstToRemove + 1 || (firstToRemove == 3 && secondToRemove == 0),
      s"Expected adjacent corners, got $firstToRemove and $secondToRemove"
    )
    val remaining = all.diff(cornersToRemove)
    if firstToRemove == 1 then remaining.reverse else remaining
