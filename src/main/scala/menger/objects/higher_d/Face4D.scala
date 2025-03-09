package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4

import scala.annotation.targetName

case class Face4D(a: Vector4, b: Vector4, c: Vector4, d: Vector4):

  lazy val normals: Seq[Vector4] = getNormals

  def asTuple: (Vector4, Vector4, Vector4, Vector4) = (a, b, c, d)
  def asSeq: Seq[Vector4] = Seq(a, b, c, d)

  def area: Float = (b - a).len() * (c - b).len()

  @targetName("plus")
  def +(delta: Vector4): Face4D = Face4D(a + delta, b + delta, c + delta, d + delta)

  @targetName("div")
  def /(scale: Float): Face4D = Face4D(a / scale, b / scale, c / scale, d / scale)

  @targetName("equals")
  def ==(that: Face4D): Boolean = asSeq == that.asSeq

  override def toString: String = faceToString(asSeq)

  private def getNormals: Seq[Vector4] =
    val edges = Seq(b - a, c - b, d - c, a - d)
    require(
      edges.map(v => v.toArray.count(_.abs > 0) == 1).forall(_ == true),
      s"Edges must be parallel to the axes, are ${edges.map(vec2string)}"
    )
    require(
      edges.sliding(2).forall({ case Seq(a, b) => a.dot(b) == 0 }),
      s"Edges must be orthogonal, are ${edges.map(vec2string)}"
    )
    normalDirections(edges).zip(normalSigns(edges)).map { case (vec, sign) => vec * sign }

  def edges: Seq[(Vector4, Vector4)] = Seq((a, b), (b, c), (c, d), (d, a))
  def plane: Plane = Plane(asSeq)

  def rotate(): Seq[Face4D] =
    logger.info(s"rotate $this")
    edges.flatMap { case (cornerA, cornerB) => rotate(cornerA, cornerB) }

  def rotate(cornerA: Vector4, cornerB: Vector4): Seq[Face4D] =
    val allCorners = Seq(a, b, c, d)
    val corners = Seq(cornerA, cornerB)

    val oppositeCorners = remainingCorners(allCorners, corners)
    logger.info(s"${corners.map(vec2string)} -> ${oppositeCorners.map(vec2string)}")
    require(
      oppositeCorners.size == 2,
      s"${corners.map(_.asString)} not in ${Set(a, b, c, d).map(_.asString)}"
    )
    val distance = oppositeCorners.head - corners.last
    require(
      distance == oppositeCorners.last - corners.head,
      s"Corners must be opposite - got ${corners.map(_.asString)} in $this}"
    )
    normals.map { normal =>
      val newOpposites = corners.map(_ + normal * distance.len())
      Face4D(a, b, newOpposites.last, newOpposites.head)
    }

def normalDirections(edgeVectors: Seq[Vector4]): Seq[Vector4] =
  /** normals point in the two directions orthogonal to the edges */
  val edgeDirectionIndices = edgeVectors.toSet.flatMap(setIndices(_))
  val normalIndices = (0 until Face4D.dimension).toSet.diff(edgeDirectionIndices)
  normalIndices.map(unitVector).toSeq

def normalSigns(edgeVectors: Seq[Vector4]): Seq[Float] =
  /**
   *  signs depend on the directions the first two edges are traversed - if the first edge is
   *  traversed from - to +, the first normal has positive sign, else negative. Analogous for
   *  the second edge and the second normal.
   */
  val firstTwoEdges = edgeVectors.take(2)
  val sum = firstTwoEdges.reduce((v1, v2) => v1 + v2).toArray.toIndexedSeq
  sum.filter(_.abs > 0).map(_.sign)

object Face4D:
  val dimension = 4
  def apply(vertices: Seq[Vector4]): Face4D =
    require(
      vertices.length == dimension,
      s"Need 4$dimension vertices, have ${vertices.length}: ${vertices.map(vec2string)}"
    )
    require(
      (vertices :+ vertices.head).sliding(2).map(edge => edge.head.dst2(edge.last)).toSet.size == 1,
      s"Vertices must all be same length, are ${vertices.map(vec2string)}"
    )
    Face4D(vertices.head, vertices(1), vertices(2), vertices(3))

def faceToString(seq: Seq[Vector4]): String = seq.map(vec2string).mkString("(", ", ", ")")

val unitVectors = Set(
  Vector4.X, Vector4.Y, Vector4.Z, Vector4.W, -Vector4.X, -Vector4.Y, -Vector4.Z, -Vector4.W
)
val positiveUnitVectors = unitVectors.map(vec => vec * vec.toArray.filter(_ != 0).sum)
def normals(vecs: Seq[Vector4]): Set[Vector4] =
  require(vecs.size == 2, s"Need 2 vectors, have ${vecs.size}: ${vecs.map(vec2string)}")
  require(unitVectors.contains(vecs.head), s"vec1 must be a unit vector, is ${vecs.head}")
  require(unitVectors.contains(vecs(1)), s"vec2 must be a unit vector, is ${vecs(1)}")
  require(vecs.head != vecs(1), s"vec1 and vec2 must be different, are ${vecs.head}")
  val normals = positiveUnitVectors.filter(vec => vec.dot(vecs.head) == 0 && vec.dot(vecs(1)) == 0)
  require(normals.size == 2, s"Expected 2 normals, have ${normals.size}: ${normals.map(vec2string)}")
  normals

def setIndices(v: Vector4, epsilon: Float = 1e-6): Seq[Int] =
  allIndicesWhere(v.toArray.toIndexedSeq, s => s.abs > epsilon)

def allIndicesWhere[A](s: Seq[A], pred: A => Boolean): Seq[Int] =
  s.zipWithIndex.filter { case (elem, _) => pred(elem) }.map { case (_, index) => index }

def unitVector(direction: Int): Vector4 =
  /** create a unit vector in the ith direction */
  require(0 until Face4D.dimension contains direction)
  val vec = new Array[Float](Face4D.dimension)
  vec(direction) = 1
  Vector4(vec)

def remainingCorners(allCorners: Seq[Vector4], cornersToRemove: Seq[Vector4]): Seq[Vector4] =
  require(allCorners.size == 4, s"Need 4 corners, have ${allCorners.size}")
  require(cornersToRemove.size == 2, s"Need 2 corners to remove, have ${cornersToRemove.size}")
  val firstToRemove = allCorners.indexOf(cornersToRemove.head)  
  val secondToRemove = allCorners.indexOf(cornersToRemove.last)
  require(
    secondToRemove == firstToRemove + 1 || (firstToRemove == 3 && secondToRemove == 0), 
    s"Expected adjacent corners, got $firstToRemove and $secondToRemove" 
  )
  val remaining = allCorners.diff(cornersToRemove)
  if firstToRemove == 1 then remaining.reverse else remaining

