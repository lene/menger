package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4

import scala.math.abs

class TesseractSponge(level: Int) extends Mesh4D:
  lazy val faces: Seq[RectVertices4D] =
    if level == 0 then Tesseract().faces else nestedFaces.flatten

  private def nestedFaces = for (
    xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1; ww <- -1 to 1
    if abs(xx) + abs(yy) + abs(zz) + abs(ww) > 2
  ) yield shrunkSubSponge.map(translate(_, Vector4(xx / 3f, yy / 3f, zz / 3f, ww / 3f)))

  private def shrunkSubSponge: Seq[RectVertices4D] =
    subSponge.map { case (a, b, c, d) => (a / 3, b / 3, c / 3, d / 3) }

  private def subSponge: Seq[RectVertices4D] = TesseractSponge(level - 1).faces

  private def translate(face: RectVertices4D, delta: Vector4): RectVertices4D =
    face match { case (a, b, c, d) => (a + delta, b + delta, c + delta, d + delta) }


class TesseractSponge2(level: Int, size: Float = 1) extends Mesh4D:
  lazy val faces: Seq[RectVertices4D] =
    if level == 0 then Tesseract(size).faces else nestedFaces.flatten

  def nestedFaces: Seq[Seq[RectVertices4D]] =
    TesseractSponge2(level - 1).faces.map(subdividedFace)

  def subdividedFace(face: RectVertices4D): Seq[RectVertices4D] =
    val corners = cornerPoints(face)
    val result = Seq(
      (corners("a"), corners("ab1"), corners("da2bc11"), corners("da2")), // 1 // top left
      (corners("ab1"), corners("ab2"), corners("da2bc12"), corners("da2bc11")), // 2 // top middle
      (corners("ab2"), corners("b"), corners("bc1"), corners("da2bc12")), // 3 // top right
      (corners("da2"), corners("da2bc11"), corners("da1bc21"), corners("da1")), // 4 // middle left
      (corners("da2bc12"), corners("bc1"), corners("bc2"), corners("da1bc22")), // 5 // middle right
      (corners("da1"), corners("da1bc21"), corners("cd2"), corners("d")), // 6 // bottom left
      (corners("da1bc21"), corners("da1bc22"), corners("cd1"), corners("cd2")), // 7 // bottom middle
      (corners("da1bc22"), corners("bc2"), corners("c"), corners("cd1")) // 8 // bottom right
    )
    result

  def cornerPoints(face: RectVertices4D): Map[String, Vector4] =
    val (a, b, c, d) = face
    val ab1 = a + (b - a) / 3
    val ab2 = a + (b - a) * 2 / 3
    val bc1 = b + (c - b) / 3
    val bc2 = b + (c - b) * 2 / 3
    val cd1 = c + (d - c) / 3
    val cd2 = c + (d - c) * 2 / 3    // reversed direction
    val da1 = d + (a - d) / 3
    val da2 = d + (a - d) * 2 / 3    // reversed direction
    val da1bc21 = da1 + (bc2 - da1) / 3
    val da1bc22 = da1 + (bc2 - da1) * 2 / 3
    val da2bc11 = da2 + (bc1 - da2) / 3
    val da2bc12 = da2 + (bc1 - da2) * 2 / 3
    Map(
      "a" -> a, "b" -> b, "c" -> c, "d" -> d,
      "ab1" -> ab1, "ab2" -> ab2, "bc1" -> bc1, "bc2" -> bc2,
      "cd1" -> cd1, "cd2" -> cd2, "da1" -> da1, "da2" -> da2,
      "da1bc21" -> da1bc21, "da1bc22" -> da1bc22, "da2bc11" -> da2bc11, "da2bc12" -> da2bc12
    )
