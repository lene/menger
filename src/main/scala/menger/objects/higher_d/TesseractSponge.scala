package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import scala.math.abs

class TesseractSponge(level: Int) extends Fractal4D(level):

  require(level >= 0, "Level must be non-negative")

  lazy val faces: Seq[Face4D] = if level == 0 then Tesseract().faces else nestedFaces.flatten

  private def nestedFaces =
    for (
      xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1; ww <- -1 to 1
      if abs(xx) + abs(yy) + abs(zz) + abs(ww) > 2
    ) yield shrunkSubSponge.map(_ + Vector4(xx / 3f, yy / 3f, zz / 3f, ww / 3f))

  private def shrunkSubSponge: Seq[Face4D] = subSponge.map { _ / 3 }

  private def subSponge: Seq[Face4D] = TesseractSponge(level - 1).faces