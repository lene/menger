package menger.objects.higher_d

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.math.Vector4
import menger.objects.*

import scala.annotation.targetName
import scala.math.abs


trait Mesh4D extends RectMesh:
  lazy val faces: Seq[RectVertices4D]

/** A tesseract of edge length `size` centered at the origin */
case class Tesseract(
  size: Float = 1.0,
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Mesh4D:

  lazy val vertices: Seq[Vector4] = for (
    xx <- -1 to 1 by 2; yy <- -1 to 1 by 2; zz <- -1 to 1 by 2; ww <- -1 to 1 by 2
  ) yield Vector4(xx * size / 2, yy * size / 2, zz * size / 2, ww * size / 2)

  // order of vertices taken from
  // https://github.com/lene/HyperspaceExplorer/blob/master/src/Displayable/Object/ObjectImplementations.cpp#L55
  lazy val faceIndices: Seq[RectIndices] = Seq(
    ( 0, 1, 3, 2), ( 0, 1, 5, 4), ( 0, 1, 9, 8), ( 0, 2, 6, 4),
    ( 0, 2,10, 8), ( 0, 4,12, 8), ( 1, 3, 7, 5), ( 1, 3,11, 9),
    ( 1, 5,13, 9), ( 2, 3, 7, 6), ( 2, 3,11,10), ( 2, 6,14,10),
    ( 3, 7,15,11), ( 4, 5, 7, 6), ( 4, 5,13,12), ( 4, 6,14,12),
    ( 5, 7,15,13), ( 6, 7,15,14), ( 8, 9,11,10), ( 8, 9,13,12),
    ( 8,10,14,12), ( 9,11,15,13), (10,11,15,14), (12,13,15,14)
  )

  lazy val faces: Seq[RectVertices4D] = faceIndices.map {
    case (a, b, c, d) => (vertices(a), vertices(b), vertices(c), vertices(d))
  }

  lazy val edges: Seq[(Vector4, Vector4)] = faces.flatMap {
    // sets instead of tuples so edges are equal regardless of direction
    case (a, b, c, d) => Seq(Set(a, b), Set(b, c), Set(c, d), Set(d, a))
  }.distinct.map(set => (set.head, set.last))

extension (v: Vector4)
  @targetName("times")
  def *(a: Float): Vector4 = Vector4(v.x * a, v.y * a, v.z * a, v.w * a)
  @targetName("dividedBy")
  def /(a: Float): Vector4 = v * (1 / a)
  @targetName("plus")
  def +(v2: Vector4): Vector4 = Vector4(v.x + v2.x, v.y + v2.y, v.z + v2.z, v.w + v2.w)


class TesseractSponge(level: Int) extends Mesh4D:
  lazy val faces: Seq[RectVertices4D] = if level == 0 then Tesseract().faces else
    val t = TesseractSponge(level - 1)
    val multipliedFaces = t.faces.map { case (a, b, c, d) => (a / 3, b / 3, c / 3, d / 3) }
    val nestedFaces = for (
      xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1; ww <- -1 to 1
      if abs(xx) + abs(yy) + abs(zz) + abs(ww) > 2
    ) yield multipliedFaces.map {
      case (a, b, c, d) =>
        val shift = Vector4(xx.toFloat, yy.toFloat, zz.toFloat, ww.toFloat) / 3
        (a + shift, b + shift, c + shift, d + shift)
    }
    nestedFaces.flatten
