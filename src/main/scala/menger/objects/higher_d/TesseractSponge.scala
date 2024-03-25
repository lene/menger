package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4

import scala.math.abs

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
