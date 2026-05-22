package menger.objects.higher_d

import menger.common.Vector


class TesseractSponge2(level: Float, size: Float = 1) extends Fractal4D(level):

  private type CornerMap = Map[String, Vector[4]]

  require(level >= 0, "Level must be non-negative")

  lazy val vertices: Seq[Vector[4]] = faces.flatMap(_.asSeq).distinct
  lazy val faces: Seq[Face4D[V]] = if level.toInt == 0 then Tesseract(size).faces else nestedFaces
  override def cells: Seq[Cell4D] = Seq.empty

  private def mergeVertices(faces: Seq[Face4D[V]], epsilon: Float = 1e-4f): Seq[Face4D[V]] =
    val canonicalMap = scala.collection.mutable.HashMap[(Long, Long, Long, Long), Vector[4]]()
    def vkey(v: Vector[4]): (Long, Long, Long, Long) =
      ((v(0) / epsilon).round, (v(1) / epsilon).round, (v(2) / epsilon).round, (v(3) / epsilon).round)
    def canonical(v: Vector[4]): Vector[4] = canonicalMap.getOrElseUpdate(vkey(v), v)
    faces.map(f => Face4D(canonical(f.a), canonical(f.b), canonical(f.c), canonical(f.d)))

  private[higher_d] def nestedFaces: Seq[Face4D[V]] =
    val prevLevel = level.toInt - 1
    val raw = TesseractSponge2(level - 1).faces.flatMap(faceGenerator)
    val filtered = raw.filter(f => f.asSeq.forall(v => isInsideKthSponge(v, prevLevel)))
    mergeVertices(filtered)

  // A vertex is inside the level-k Menger sponge iff, at each scale level 1..k,
  // the number of coordinates with digit==1 (in base-3 decomposition) is at most 1.
  // Boundary vertices are treated as valid: if the vertex is within eps of a sub-cube
  // boundary, both adjacent sub-cube digits are tried (vertex is valid if any combination
  // has count <= 1).
  private def isInsideKthSponge(v: Vector[4], k: Int, eps: Float = 1e-4f): Boolean =
    (1 to k).forall { lvl =>
      val scale = math.pow(3.0, lvl)
      val epsScaled = eps.toDouble * scale
      val possibleDigits = (0 until 4).map { dim =>
        val x = (v(dim) + 0.5f).toDouble * scale
        val fl = math.floor(x).toInt
        val frac = x - fl
        if frac < epsScaled then
          Seq(((fl - 1) % 3 + 3) % 3, fl % 3).distinct
        else if (1.0 - frac) < epsScaled then
          Seq(fl % 3, (fl + 1) % 3).distinct
        else
          Seq(fl % 3)
      }
      def anyValidCombo(dim: Int, count1: Int): Boolean =
        if count1 >= 2 then false
        else if dim == 4 then true
        else possibleDigits(dim).exists(d => anyValidCombo(dim + 1, if d == 1 then count1 + 1 else count1))
      anyValidCombo(0, 0)
    }

  private[higher_d] def faceGenerator(face: Face4D[V]): Seq[Face4D[V]] =
    generateFlatParts(face) ++ generatePerpendicularParts(face)

  private[higher_d] def generateFlatParts(face: Face4D[V]): Seq[Face4D[V]] =
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

  private[higher_d] def generatePerpendicularParts(face: Face4D[V]): Seq[Face4D[V]] =
    val c = cornerPoints(face)
    val centralPart = Face4D(c("da2bc11"), c("da2bc12"), c("da1bc22"), c("da1bc21"))
    centralPart.extrude()

  private[higher_d] def cornerPoints(face: Face4D[V]): CornerMap =
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
