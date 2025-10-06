package menger.objects.higher_d

import menger.objects.Vector
import org.scalatest.Inspectors.forAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.*
import CustomMatchers.*
import menger.Const

class TesseractSponge2Suite extends AnyFlatSpec with RectMesh with Matchers:

  trait Sponge2:
    val tesseract: Tesseract = Tesseract(2)
    val sponge2: TesseractSponge2 = TesseractSponge2(1)
    val face: Face4D = tesseract.faces.head
    require(
      face == Face4D(Vector[4](-1, -1, -1, -1), Vector[4](-1, -1, -1, 1), Vector[4](-1, -1, 1, 1), Vector[4](-1, -1, 1, -1))
    )
    val subfaces: Seq[Face4D] = sponge2.faceGenerator(face)
    val centerSubface: List[Vector[4]] = List(
      Vector[4](-1, -1, -1/3f, 1/3f), Vector[4](-1, -1, 1/3f, 1/3f),
      Vector[4](-1, -1, 1/3f, -1/3f), Vector[4](-1, -1, -1/3f, -1/3f)
    )
    val centerSubfaceEdges: Seq[(Vector[4], Vector[4])] = List(
      (centerSubface(0), centerSubface(1)), (centerSubface(1), centerSubface(2)),
      (centerSubface(2), centerSubface(3)), (centerSubface(3), centerSubface(0))
    )
    val subfacesString: String = subfaces.toString.replace("),", "),\n")
    val flatSubfaces: Seq[Face4D] = sponge2.generateFlatParts(face)
    val flatSubfacesString: String = flatSubfaces.toString.replace("),", "),\n")
    val perpendicularSubfaces: Seq[Face4D] = sponge2.generatePerpendicularParts(face)
    val perpendicularSubfacesString: String =
      perpendicularSubfaces.map(rect2str).mkString(", ").replace("),", "),\n")

    def diffToFaces(faces: Seq[Face4D], face2: List[Vector[4]]): String =
      def diffBetweenFaces(face1: List[Vector[4]], face2: List[Vector[4]]): List[Vector[4]] =
        face1.zip(face2).map((vertex1, vertex2) => vertex2 - vertex1)
      val facesAsList: Seq[List[Vector[4]]] = faces.map(_.asSeq.toList)
      facesAsList.map(
        face1 => diffBetweenFaces(face1, face2).map(_.toString)
      ).toString.replace("),", "),\n").replace("Vector(", "Vector(\n ")

    def lineRoughlyEquals(line1: (Vector[4], Vector[4]), line2: (Vector[4], Vector[4])): Boolean =
      line1._1 === line2._1 && line1._2 === line2._2

  def rect2str(rect: Face4D): String = rect.asSeq.map(_.toString).mkString("(", ", ", ")")

  def seq2str(seq: Seq[Face4D]): String = seq.map(rect2str).mkString(",\n")

  def sponge2str(sponge: TesseractSponge2): String = seq2str(sponge.faces) + "\n"

  "A TesseractSponge2 level 0" should "have 24 faces" in:
    TesseractSponge2(0).faces should have size 24

  "A TesseractSponge level < 0" should "be imposssible" in:
    an[IllegalArgumentException] should be thrownBy {TesseractSponge2(-1)}

  "A TesseractSponge2 level 1" should "have 16 * 24 faces" in:
    TesseractSponge2(1).faces should have size (16 * 24)

  "A TesseractSponge2 level 2" should "have 16 * 16 * 24 faces" in :
    TesseractSponge2(2).faces should have size (16 * 16 * 24)

  "A subdivided face's corner points" should "contain the original face's corners" in new Sponge2:
    forAll (face.asSeq) { v => sponge2.cornerPoints(face).values should containEpsilon (v) }

  it should "contain the interior points at a distance of a third from the edge" in new Sponge2:
    forAll (Seq(-1/3f, 1/3f)) { z =>
      forAll(Seq(-1/3f, 1/3f)) { w =>
        sponge2.cornerPoints(face).values should containEpsilon (Vector[4](-1, -1, z, w))
      }
    }

  it should "contain 16 points" in new Sponge2:
    sponge2.cornerPoints(face) should have size 16

  it should "contain 16 distinct points" in new Sponge2:
    sponge2.cornerPoints(face).values.toSet should have size 16

  "A subdivided face's flat parts" should "have size 8" in new Sponge2:
    flatSubfaces should have size 8

  it should "have 8 distinct surfaces" in new Sponge2:
    flatSubfaces.toSet should have size 8

  it should "contain top left subface" in new Sponge2:
    withClue(flatSubfacesString) {
      flatSubfaces should containAllEpsilon(List(
          Vector[4](-1, -1, -1,    -1),    Vector[4](-1, -1, -1,    -1/3f),
          Vector[4](-1, -1, -1/3f, -1/3f), Vector[4](-1, -1, -1/3f, -1)
        ))
    }

  it should "contain top middle subface" in new Sponge2:
    withClue(subfacesString) {
      flatSubfaces should containAllEpsilon(List(
          Vector[4](-1, -1, -1,    -1/3f), Vector[4](-1, -1, -1,     1/3f),
          Vector[4](-1, -1, -1/3f,  1/3f), Vector[4](-1, -1, -1/3f, -1/3f)
        ))
    }

  it should "contain top right subface" in new Sponge2:
    withClue(subfacesString) {
      flatSubfaces should containAllEpsilon(List(
          Vector[4](-1, -1, -1,    1/3f), Vector[4](-1, -1, -1,    1),
          Vector[4](-1, -1, -1/3f, 1),    Vector[4](-1, -1, -1/3f, 1/3f)
        ))
    }

  it should "contain middle left subface" in new Sponge2:
    withClue(subfacesString) {
      flatSubfaces should containAllEpsilon(List(
          Vector[4](-1, -1, -1/3f, -1),    Vector[4](-1, -1, -1/3f, -1/3f),
          Vector[4](-1, -1,  1/3f, -1/3f), Vector[4](-1, -1,  1/3f, -1)
        ))
    }

  it should "not contain center subface" in new Sponge2:
    withClue(subfacesString) { flatSubfaces should not (containAllEpsilon (centerSubface)) }

  it should "contain middle right subface" in new Sponge2:
    withClue(subfacesString) {
      flatSubfaces should containAllEpsilon(List(
          Vector[4](-1, -1, -1/3f, 1/3f), Vector[4](-1, -1, -1/3f, 1),
          Vector[4](-1, -1,  1/3f, 1),    Vector[4](-1, -1,  1/3f, 1/3f)
        ))
    }

  it should "contain bottom left subface" in new Sponge2:
    withClue(subfacesString) {
      flatSubfaces should containAllEpsilon(List(
          Vector[4](-1, -1, 1/3f, -1/3f), Vector[4](-1, -1, 1/3f, -1),
          Vector[4](-1, -1, 1,    -1),    Vector[4](-1, -1, 1,    -1/3f)
        ))
    }

  it should "contain bottom middle subface" in new Sponge2:
    withClue(subfacesString) {
      flatSubfaces should containAllEpsilon(List(
          Vector[4](-1, -1, 1/3f,  1/3f), Vector[4](-1, -1, 1,     1/3f),
          Vector[4](-1, -1, 1,    -1/3f), Vector[4](-1, -1, 1/3f, -1/3f)
        ))
    }

  it should "contain bottom right subface" in new Sponge2:
    withClue(subfacesString) {
      flatSubfaces should containAllEpsilon(List(
          Vector[4](-1, -1, 1/3f, 1/3f), Vector[4](-1, -1, 1/3f, 1),
          Vector[4](-1, -1, 1,    1),    Vector[4](-1, -1, 1,    1/3f)
        ))
    }

  it should "all have 1/9 the original area" in new Sponge2:
    val subfaceArea: Seq[Float] = flatSubfaces.map(_.area)
    forAll (subfaceArea) { _ shouldBe face.area / 9 +- Const.epsilon }

  "A subdivided face's perpendicular parts" should "have size 8" in new Sponge2:
    perpendicularSubfaces should have size 8

  it should "have 8 distinct surfaces" in new Sponge2:
    perpendicularSubfaces.toSet should have size 8

  it should "not contain any of the flat surfaces" in new Sponge2:
    flatSubfaces.toSet.intersect(perpendicularSubfaces.toSet) shouldBe empty

  it should "all have the same base lines" in new Sponge2:
    forAll (perpendicularSubfaces) { subface =>
      centerSubfaceEdges.exists(line => lineRoughlyEquals(line, (subface.a, subface.b))) shouldBe true
    }

  it should "all be 1/3 of the original side length long" in new Sponge2:
    perpendicularSubfaces.foreach(subface =>
      (subface.b - subface.a).len shouldBe 2/3f +- Const.epsilon
    )

  it should "all be parallel to the axes" in new Sponge2:
    def isParallelToAxes(v: Vector[4]): Boolean =
      v.count(f => math.abs(f) < Const.epsilon) == 3

    forAll(perpendicularSubfaces) { face =>
      val edges = Seq(face.b - face.a, face.c - face.b, face.d - face.c, face.a - face.d)
      forAll(edges) { edge => isParallelToAxes(edge) shouldBe true }
    }

  it should "all have 1/9 the original area" in new Sponge2:
    val originalArea: Float = face.area
    val subfaceArea: Seq[Float] = perpendicularSubfaces.map(_.area)
    subfaceArea.foreach(_ shouldBe originalArea / 9 +- Const.epsilon)

  it should "create faces perpendicular to the original face" in:
    val sponge = TesseractSponge2(1)
    val face = Tesseract().faces.head
    val perpFaces = sponge.generatePerpendicularParts(face)

    // Each perpendicular face should share a plane with the original face's normal
    forAll(perpFaces) { perpFace =>
      val origNormals = face.normals.toSet
      perpFace.plane.units should contain atLeastOneOf(origNormals.head, origNormals.last)
    }

  it should "contain face rotated into y direction" in new Sponge2:
    val expected = List(
      Vector[4](-1, -1,    -1/3f, -1/3f), Vector[4](-1, -1/3f, -1/3f, -1/3f),
      Vector[4](-1, -1/3f, -1/3f,  1/3f), Vector[4](-1, -1,    -1/3f,  1/3f)
    )
    val clue = s"""\nexpected:
  ${faceToString(expected)}
actual:
 $perpendicularSubfacesString
diff: ${diffToFaces(perpendicularSubfaces, expected)}\n"""
    withClue(clue) { perpendicularSubfaces should containAllEpsilon (expected) }

  "A subdivided face" should "contain top left subface" in new Sponge2:
    withClue(subfacesString) {
      subfaces should containAllEpsilon(List(
        Vector[4](-1, -1, -1,    -1),    Vector[4](-1, -1, -1,    -1/3f),
        Vector[4](-1, -1, -1/3f, -1/3f), Vector[4](-1, -1, -1/3f, -1)
      ))
    }

  it should "contain top middle subface" in new Sponge2:
    withClue(subfacesString) {
      subfaces should containAllEpsilon(List(
          Vector[4](-1, -1, -1,    -1/3f), Vector[4](-1, -1, -1,     1/3f),
          Vector[4](-1, -1, -1/3f,  1/3f), Vector[4](-1, -1, -1/3f, -1/3f)
        ))
    }

  it should "contain top right subface" in new Sponge2:
    withClue(subfacesString) {
      subfaces should containAllEpsilon(List(
          Vector[4](-1, -1, -1,    1/3f), Vector[4](-1, -1, -1,    1),
          Vector[4](-1, -1, -1/3f, 1),    Vector[4](-1, -1, -1/3f, 1/3f)
        ))
    }

  it should "contain middle left subface" in new Sponge2:
    withClue(subfacesString) {
      subfaces should containAllEpsilon(List(
          Vector[4](-1, -1, -1/3f, -1),    Vector[4](-1, -1, -1/3f, -1/3f),
          Vector[4](-1, -1,  1/3f, -1/3f), Vector[4](-1, -1,  1/3f, -1)
        ))
    }

  it should "not contain center subface" in new Sponge2:
    withClue(subfacesString) { subfaces should not (containAllEpsilon(centerSubface)) }

  it should "contain middle right subface" in new Sponge2:
    withClue(subfacesString) {
      subfaces should containAllEpsilon(List(
          Vector[4](-1, -1, -1/3f, 1/3f), Vector[4](-1, -1, -1/3f, 1),
          Vector[4](-1, -1,  1/3f, 1),    Vector[4](-1, -1,  1/3f, 1/3f)
        ))
    }

  it should "contain bottom left subface" in new Sponge2:
    withClue(subfacesString) {
      subfaces should containAllEpsilon(List(
          Vector[4](-1, -1, 1/3f, -1/3f), Vector[4](-1, -1, 1/3f, -1),
          Vector[4](-1, -1, 1,    -1),    Vector[4](-1, -1, 1,    -1/3f)
        ))
    }

  it should "contain bottom middle subface" in new Sponge2:
    withClue(subfacesString) {
      subfaces should containAllEpsilon(List(
          Vector[4](-1, -1, 1/3f,  1/3f), Vector[4](-1, -1, 1,     1/3f),
          Vector[4](-1, -1, 1,    -1/3f), Vector[4](-1, -1, 1/3f, -1/3f)
        ))
    }

  it should "contain bottom right subface" in new Sponge2:
    withClue(subfacesString) {
      subfaces should containAllEpsilon(List(
          Vector[4](-1, -1, 1/3f, 1/3f), Vector[4](-1, -1, 1/3f, 1),
          Vector[4](-1, -1, 1,    1),    Vector[4](-1, -1, 1,    1/3f)
        ))
    }

  Seq(Plane.xw, Plane.yw, Plane.xz, Plane.yz).foreach { plane =>
    it should s"contain Face4D in $plane bordered on center hole" in new Sponge2:
      withClue(s"$subfacesString\n${subfaces.map(_.plane)}") {
        subfaces.map(_.plane) should contain(plane)
      }
  }

  it should "contain 16 subfaces" in new Sponge2:
    subfaces should have size 16

  it should "contain 16 distinct subfaces" in new Sponge2:
    subfaces.toSet should have size 16

  "nested faces" should "correctly recursively subdivide all faces" in:
    val level1 = TesseractSponge2(1)
    val level0 = TesseractSponge2(0)
    val nestedCount = level0.faces.map(face => level1.faceGenerator(face).size).sum
    level1.faces should have size nestedCount

  "All faces in TesseractSponge2" should "be parallel to an axis" in new Sponge2:
    forAll(subfaces) { face =>
      val edgeVectors = face.edges.map(_.diff)
      forAll(edgeVectors) { edge => edge.count(v => math.abs(v) < Const.epsilon) shouldBe 3 }
    }

  it should "have opposite edges parallel, equal length and opposite direction" in new Sponge2:
    forAll(subfaces) { face =>
      val edgeVectors = face.edges.map(_.diff)
      (edgeVectors(0) + edgeVectors(2)).len shouldBe 0f +- Const.epsilon
      (edgeVectors(1) + edgeVectors(3)).len shouldBe 0f +- Const.epsilon
    }

  "Each subdivision" should "multiply surface area by 16/9" in:
    val level0Area = TesseractSponge2(0).faces.map(_.area).sum
    val level1Area = TesseractSponge2(1).faces.map(_.area).sum
    (level1Area / level0Area) shouldBe 16.0f / 9.0f +- Const.epsilon

  private def round(f: Float): Float = math.round(f / Const.epsilon) * Const.epsilon

  "All level 1 sponge corner points" should "have absolute coordinate values 1/2 or 1/6" in new Sponge2:
    val cornerPoints: Seq[Vector[4]] = sponge2.faces.flatMap(_.asSeq)
    val cornerCoordinateValues: Set[Float] = cornerPoints.toSet.flatMap(_.v).map(math.abs).map(round)
    val clue: String = sponge2str(sponge2).replace("0.83", s"${Console.YELLOW}0.83${Console.RED}")
    withClue(clue) {cornerCoordinateValues should contain only (round(1/2f), round(1/6f))}

  "Subdivided subfaces" should "have absolute coordinate values 1/2 or 1/6 in face 1" in:
    checkCoordinatesOfSubdividedSpongeFace(TesseractSponge2(0), 0)

  for i <- 1 to 23 do
    it should s"have absolute coordinate values 1/2 or 1/6 in face ${i+1}" in:
      checkCoordinatesOfSubdividedSpongeFace(TesseractSponge2(0), i)

  "toString" should "return the class name" in new Sponge2:
    sponge2.toString should include("TesseractSponge2")

  it should "contain the sponge level" in new Sponge2:
    sponge2.toString should include(s"level=${menger.objects.float2string(sponge2.level)}")

  it should "contain the number of faces" in new Sponge2:
    sponge2.toString should include(s"${sponge2.faces.size} faces")

  private def checkCoordinatesOfSubdividedSpongeFace(sponge: TesseractSponge2, i: Int) =
    val subface: Face4D = sponge.faces(i)
    val subdividedFace: Seq[Face4D] = sponge.faceGenerator(subface)
    val cornerPoints: Seq[Vector[4]] = subdividedFace.flatMap(_.asSeq)
    val cornerCoordinateValues: Set[Float] = cornerPoints.toSet.flatMap(_.v).map(math.abs).map(round)
    val coords = seq2str(subdividedFace).replace("0.83", s"${Console.YELLOW}0.83${Console.RED}")
    val clue = s"${i+1}/${sponge.faces.length} (${Plane(subface)})\n$coords\n"
    withClue(clue) {cornerCoordinateValues should contain atLeastOneOf(round(1 / 2f), round(1 / 6f))}
    withClue(clue) {cornerCoordinateValues diff Set(round(1 / 2f), round(1 / 6f)) shouldBe empty}

  "fractional level 0.5" should "instantiate" in:
    val sponge = TesseractSponge2(0.5f)
    sponge.level shouldBe 0.5f

  it should "use floor for face generation" in:
    val sponge = TesseractSponge2(0.5f)
    val level0 = TesseractSponge2(0f)
    sponge.faces should have size level0.faces.size
