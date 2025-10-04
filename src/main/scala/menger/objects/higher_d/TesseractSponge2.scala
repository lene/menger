package menger.objects.higher_d

import com.typesafe.scalalogging.LazyLogging
import menger.objects.Vector

/**
 * Represents a 4D Menger sponge based on a tesseract.
 * Each level is constructed by subdividing each face of the previous level into 9 smaller squares,
 * and replacing the center one with faces flipped perpendicular to it.
 * @param level The recursion level (0 = regular tesseract)
 * @param size The size of the bounding tesseract
 */
class TesseractSponge2(level: Float, size: Float = 1) extends Fractal4D(level) with LazyLogging:

  private type CornerMap = Map[String, Vector[4]]

  require(level >= 0, "Level must be non-negative")

  lazy val faces: Seq[Face4D] = if level.toInt == 0 then Tesseract(size).faces else nestedFaces

  private[higher_d] def nestedFaces: Seq[Face4D] =
    TesseractSponge2(level - 1).faces.flatMap(faceGenerator)

  private[higher_d] def faceGenerator(face: Face4D): Seq[Face4D] =
    generateFlatParts(face) ++ generatePerpendicularParts(face)

  private[higher_d] def generateFlatParts(face: Face4D): Seq[Face4D] =
    // split the face into 9 smaller squares and return all except the center one
    val c = cornerPoints(face)
    Seq(
      Face4D(c("a"), c("ab1"), c("da2bc11"), c("da2")), // 1 // top left
      Face4D(c("ab1"), c("ab2"), c("da2bc12"), c("da2bc11")), // 2 // top middle
      Face4D(c("ab2"), c("b"), c("bc1"), c("da2bc12")), // 3 // top right
      Face4D(c("da2"), c("da2bc11"), c("da1bc21"), c("da1")), // 4 // middle left
      Face4D(c("da2bc12"), c("bc1"), c("bc2"), c("da1bc22")), // 5 // middle right
      Face4D(c("da1"), c("da1bc21"), c("cd2"), c("d")), // 6 // bottom left
      Face4D(c("da1bc21"), c("da1bc22"), c("cd1"), c("cd2")), // 7 // bottom middle
      Face4D(c("da1bc22"), c("bc2"), c("c"), c("cd1")) // 8 // bottom right
    )

  private[higher_d] def generatePerpendicularParts(face: Face4D): Seq[Face4D] =
    val c = cornerPoints(face)
    val centralPart = Face4D(c("da2bc11"), c("da2bc12"), c("da1bc22"), c("da1bc21"))
    centralPart.rotate()

  private[higher_d] def cornerPoints(face: Face4D): CornerMap =
    val (a, b, c, d) = face.asTuple

    val edgeAB = (b - a) / 3
    val edgeBC = (c - b) / 3
    val edgeCD = (d - c) / 3
    val edgeDA = (a - d) / 3

    val ab1 = a + edgeAB
    val ab2 = a + edgeAB * 2
    val bc1 = b + edgeBC
    val bc2 = b + edgeBC * 2
    val cd1 = c + edgeCD
    val cd2 = c + edgeCD * 2    // reversed direction
    val da1 = d + edgeDA
    val da2 = d + edgeDA * 2
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
