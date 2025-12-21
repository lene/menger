package menger.objects.higher_d

import scala.annotation.targetName

import menger.common.Const
import menger.common.Vector


case class Face4D(a: Vector[4], b: Vector[4], c: Vector[4], d: Vector[4]):

  lazy val normals: Seq[Vector[4]] = getNormals

  def asTuple: (Vector[4], Vector[4], Vector[4], Vector[4]) = (a, b, c, d)
  def asSeq: Seq[Vector[4]] = Seq(a, b, c, d)

  def area: Float = (b - a).len * (c - b).len

  @targetName("plus")
  def +(delta: Vector[4]): Face4D = Face4D(a + delta, b + delta, c + delta, d + delta)

  @targetName("div")
  def /(scale: Float): Face4D = Face4D(a / scale, b / scale, c / scale, d / scale)

  @targetName("equals")
  def ==(that: Face4D): Boolean = asSeq == that.asSeq

  override def toString: String = asSeq.map(_.toString).mkString("(", ", ", ")")

  private def getNormals: Seq[Vector[4]] =
    val edges = Seq(b - a, c - b, d - c, a - d)
    require(
      edges.map(v => v.count(_.abs > Const.epsilon) == 1).forall(_ == true),
      s"Edges must be parallel to the axes, are ${edges.map(_.toString)}"
    )
    require(
      edges.sliding(2).forall({ case Seq(a, b) => a * b < Const.epsilon }),
      s"Edges must be orthogonal, are ${edges.map(_.toString)}"
    )
    normalDirections(edges).zip(normalSigns(edges.take(2))).map { case (vec, sign) => vec * sign }

  def edges: Seq[Edge] = Seq(Edge(a, b), Edge(b, c), Edge(c, d), Edge(d, a))
  def plane: Plane = Plane(asSeq)

  def rotate(): Seq[Face4D] =
    edges.flatMap { edge => rotate(edge) }

  def rotate(edge: Edge): Seq[Face4D] =
    val allCorners = asSeq
    val corners = edge.asSeq
    val oppositeCorners = remainingCorners(allCorners, corners)
    require(
      oppositeCorners.size == 2,
      s"${corners.map(_.toString)} not in ${Set(a, b, c, d).map(_.toString)}"
    )
    val distance = oppositeCorners.head - corners.last
    require(
      distance.dst(oppositeCorners.last - corners.head) < Const.epsilon,
      s"Corners must be opposite - got ${corners.map(_.toString)} in $this}"
    )
    normals.map { normal =>
      val newOpposites = corners.map(_ + normal * distance.len)
      Face4D(edge.v0, edge.v1, newOpposites.last, newOpposites.head)
    }


def normalDirections(edgeVectors: Seq[Vector[4]]): Seq[Vector[4]] =
  val edgeDirectionIndices = edgeVectors.toSet.flatMap(vectorIndices)
  val normalIndices = (0 until Face4D.numVertices).toSet.diff(edgeDirectionIndices)
  normalIndices.map(Vector.unit[4]).toSeq

def vectorIndices(v: Vector[4]): Seq[Int] =
  allIndicesWhere(v.toIndexedSeq, s => s.abs > Const.epsilon)


def normalSigns(edgeVectors: Seq[Vector[4]]): Seq[Float] =
  val sum = edgeVectors.reduce(_ + _)
  sum.filter(_.abs > 0).map(_.sign)

object Face4D:
  private val VERTICES_PER_FACE = 4
  val numVertices: Int = VERTICES_PER_FACE
  val dimension = 4
  def apply(vertices: Seq[Vector[4]]): Face4D =
    require(
      vertices.length == numVertices,
      s"Need $numVertices vertices, have ${vertices.length}: ${vertices.map(_.toString)}"
    )
    val squaredEdgeLengths = (vertices :+ vertices.head).sliding(2).map(
      edge => edge.head.dst2(edge.last)
    ).toSet
    require(
      squaredEdgeLengths.max - squaredEdgeLengths.min <= Const.epsilon,
      s"Vertices must all be same length, are $squaredEdgeLengths (${vertices.map(_.toString)})"
    )
    Face4D(vertices.head, vertices(1), vertices(2), vertices(3))

def allIndicesWhere[A](s: Seq[A], pred: A => Boolean): Seq[Int] =
  s.zipWithIndex.filter { case (elem, _) => pred(elem) }.map { case (_, index) => index }

def remainingCorners(allCorners: Seq[Vector[4]], cornersToRemove: Seq[Vector[4]]): Seq[Vector[4]] =
  require(allCorners.size == 4, s"Need 4 corners, have ${allCorners.size}")
  require(cornersToRemove.size == 2, s"Need 2 corners to remove, have ${cornersToRemove.size}")
  val firstToRemove = allCorners.indexWhere(_.dst(cornersToRemove.head) < Const.epsilon)
  val secondToRemove = allCorners.indexWhere(_.dst(cornersToRemove.last) < Const.epsilon)
  require(
    secondToRemove == firstToRemove + 1 || (firstToRemove == 3 && secondToRemove == 0), 
    s"Expected adjacent corners, got $firstToRemove and $secondToRemove" 
  )
  val remaining = allCorners.diff(cornersToRemove)
  // if removing middle two corners, the remaining corners are wrapped, i.e. effectively swapped
  if firstToRemove == 1 then remaining.reverse else remaining

