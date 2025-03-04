package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4

import scala.math.abs

class TesseractSponge(level: Int) extends Mesh4D:
  require(level >= 0, "Level must be non-negative")
  lazy val faces: Seq[RectVertices4D] =
    if level == 0 then Tesseract().faces else nestedFaces.flatten

  private def nestedFaces = for (
    xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1; ww <- -1 to 1
    if abs(xx) + abs(yy) + abs(zz) + abs(ww) > 2
  ) yield shrunkSubSponge.map(translate(_, Vector4(xx / 3f, yy / 3f, zz / 3f, ww / 3f)))

  private def shrunkSubSponge: Seq[RectVertices4D] = subSponge.map { face => face / 3 }

  private def subSponge: Seq[RectVertices4D] = TesseractSponge(level - 1).faces

  private def translate(face: RectVertices4D, delta: Vector4): RectVertices4D = face + delta
