package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import com.typesafe.scalalogging.Logger

class TesseractSponge2(level: Int, size: Float = 1) extends Mesh4D:
  require(level >= 0, "Level must be non-negative")
  private val logger = Logger("TesseractSponge2")
  lazy val faces: Seq[Face4D] =
    if level == 0 then Tesseract(size).faces else nestedFaces.flatten

  def nestedFaces: Seq[Seq[Face4D]] =
    TesseractSponge2(level - 1).faces.map(subdividedFace)

  def subdividedFace(face: Face4D): Seq[Face4D] =
    subdivideFlatParts(face) ++ subdividePerpendicularParts(face)

  def subdivideFlatParts(face: Face4D): Seq[Face4D] =
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

  def subdividePerpendicularParts(face: Face4D): Seq[Face4D] =
    // for each edge of the central part of the face:
    // 1. rotate the opposite vertex around the edge in the first normal direction of the face
    // 2. rotate the opposite vertex around the edge in the other normal direction of the face
    val c = cornerPoints(face)
    val centralPart = (c("da2bc11"), c("da2bc12"), c("da1bc22"), c("da1bc21"))
    val edges = Seq(
      (centralPart(0), centralPart(1)), (centralPart(1), centralPart(2)),
      (centralPart(2), centralPart(3)), (centralPart(3), centralPart(0))
    )
    val oppositeEdges = edges.drop(2) ++ edges.take(2)
    logger.debug(s"edges: ${edges.map(edgeToString).mkString("(\n", ",\n", ")")}")
    val rotated = for rotationDirection <- 0 to 1 yield
      for edge <- edges.indices yield
        rotatedRect(face, edges, oppositeEdges, edge, rotationDirection)
    rotated.flatten    

  private def rotatedRect(
    face: Face4D,
    edges: Seq[(Vector4, Vector4)], oppositeEdges: Seq[(Vector4, Vector4)],
    i: Int, j: Int
  ) =
    val rect = Face4D(
      edges(i)(0), edges(i)(1),
      Rotate(Plane(face), edges(i), edges(i)(0), 90)(j)(oppositeEdges(i)(0)),
      Rotate(Plane(face), edges(i), edges(i)(1), 90)(j)(oppositeEdges(i)(1))
    )
    logger.debug(s"original: ${faceToString(Seq(edges(i)(0), edges(i)(1), oppositeEdges(i)(0), oppositeEdges(i)(1)))}")
    logger.debug(s"plane:    ${Plane(face)}")
    logger.debug(s"rotated:  $rect")
    rect

  def cornerPoints(face: Face4D): Map[String, Vector4] =
    val (a, b, c, d) = face.asTuple
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
