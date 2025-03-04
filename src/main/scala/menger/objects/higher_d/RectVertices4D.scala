package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4

import scala.annotation.targetName

case class RectVertices4D(a: Vector4, b: Vector4, c: Vector4, d: Vector4):
  def asTuple: (Vector4, Vector4, Vector4, Vector4) = (a, b, c, d)
  def asSeq: Seq[Vector4] = Seq(a, b, c, d)
  def area: Float = (b - a).len() * (c - b).len()
  @targetName("plus")
  def +(delta: Vector4): RectVertices4D = RectVertices4D(a + delta, b + delta, c + delta, d + delta)
  @targetName("div")
  def /(scale: Float): RectVertices4D = RectVertices4D(a / scale, b / scale, c / scale, d / scale)
  @targetName("equals")
  def ==(that: RectVertices4D): Boolean = asSeq == that.asSeq


object RectVertices4D:
  def apply(seq: Seq[Vector4]): RectVertices4D =
    require(seq.length == 4, s"Need 4 vertices, have ${seq.length}: ${seq.map(vec2string)}")
    RectVertices4D(seq.head, seq(1), seq(2), seq(3))

  def apply(center: Vector4, scale: Float, normalVectors: Seq[Vector4]): RectVertices4D =
    require(normalVectors.size == 2, s"Need 2 normals, have ${normalVectors.size}: ${normalVectors.map(vec2string)}")
    val parallels = normals(normalVectors)
    RectVertices4D(
      center - parallels.head * scale - parallels.last * scale,
      center - parallels.head * scale + parallels.last * scale,
      center + parallels.head * scale + parallels.last * scale,
      center + parallels.head * scale - parallels.last * scale
    )

def faceToString(face: Seq[Vector4]): String = face.map(vec2string).mkString("(", ", ", ")")
def faceToString(face: (Vector4, Vector4, Vector4, Vector4)): String =
  faceToString(Seq(face._1, face._2, face._3, face._4))
def faceToString(face: RectVertices4D): String = faceToString(face.asSeq)

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
