package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should._

class TesseractSponge2Suite extends AnyFlatSpec with RectMesh with Matchers:

  trait Sponge2:
    val tesseract: Tesseract = Tesseract(2)
    val sponge2: TesseractSponge2 = TesseractSponge2(1)
    val face: Face4D = tesseract.faces.head
    assert(face == Face4D(
      Vector4(-1, -1, -1, -1), Vector4(-1, -1, -1, 1), Vector4(-1, -1, 1, 1), Vector4(-1, -1, 1, -1))
    )
    val subfaces: Seq[Face4D] = sponge2.subdividedFace(face)
    val centerSubface: List[Vector4] = List(
      Vector4(-1f, -1f, -1 / 3f, 1 / 3f), Vector4(-1f, -1f, 1 / 3f, 1 / 3f),
      Vector4(-1f, -1f, 1 / 3f, -1 / 3f), Vector4(-1f, -1f, -1 / 3f, -1 / 3f)
    )
    val centerSubfaceEdges: Seq[(Vector4, Vector4)] = List(
      (centerSubface(0), centerSubface(1)), (centerSubface(1), centerSubface(2)),
      (centerSubface(2), centerSubface(3)), (centerSubface(3), centerSubface(0))
    )
    val subfacesString: String = subfaces.toString.replace("),", "),\n")
    val flatSubfaces: Seq[Face4D] = sponge2.subdivideFlatParts(face)
    val perpendicularSubfaces: Seq[Face4D] = sponge2.subdividePerpendicularParts(face)
    val perpendicularSubfacesString: String =
      perpendicularSubfaces.map(rect2str).mkString(", ").replace("),", "),\n")

    def diffToFaces(faces: Seq[Face4D], face2: List[Vector4]): String =
      def diffBetweenFaces(face1: List[Vector4], face2: List[Vector4]): List[Vector4] =
        face1.zip(face2).map((vertex1, vertex2) => vertex2 - vertex1)
      val facesAsList: Seq[List[Vector4]] = faces.map(_.asSeq.toList)
      facesAsList.map(
        face1 => diffBetweenFaces(face1, face2).map(vec2string)
      ).toString.replace("),", "),\n").replace("Vector(", "Vector(\n ")

    def lineRoughlyEquals(line1: (Vector4, Vector4), line2: (Vector4, Vector4)): Boolean =
      line1._1.epsilonEquals(line2._1) && line1._2.epsilonEquals(line2._2)

  val epsilon: Float = 1e-5f

  def rect2str(rect: Face4D): String = rect.asSeq.map(vec2string).mkString("(", ", ", ")")

  def seq2str(seq: Seq[Face4D]): String = seq.map(rect2str).mkString(",\n")

  def sponge2str(sponge: TesseractSponge2): String = seq2str(sponge.faces) + "\n"

  "A TesseractSponge2 level 0" should "have 24 faces" in:
    assert(TesseractSponge2(0).faces.size == 24)

  "A TesseractSponge level < 0" should "be imposssible" in:
    assertThrows[IllegalArgumentException] {TesseractSponge2(-1)}

  "A subdivided face's corner points" should "contain the original face's corners" in new Sponge2:
    for v <- face.asSeq do assertContainsEpsilon(sponge2.cornerPoints(face).values, v)

  it should "contain the interior points at a distance of a third from the edge" in new Sponge2:
    for z <- Seq(-1/3f, 1/3f) do
      for w <- Seq(-1/3f, 1/3f) do
        val v = Vector4(-1, -1, z, w)
        assertContainsEpsilon(sponge2.cornerPoints(face).values, v)

  it should "contain 16 points" in new Sponge2:
    assert(sponge2.cornerPoints(face).size == 16)

  it should "contain 16 distinct points" in new Sponge2:
    assert(sponge2.cornerPoints(face).values.toSet.size == 16)

  ignore should "print the points" in new Sponge2:
    assert(false, sponge2.cornerPoints(face).toSeq.sortBy(v => v._2.z*10 + v._2.w).mkString("\n"))

  "A subdivided face's flat parts" should "have size 8" in new Sponge2:
    assert(flatSubfaces.size == 8)

  it should "have 8 distinct surfaces" in new Sponge2:
    assert(flatSubfaces.toSet.size == 8)

  it should "contain top left subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        flatSubfaces, List(
          Vector4(-1f, -1f, -1f, -1f),
          Vector4(-1f, -1f, -1f, -1 / 3f),
          Vector4(-1f, -1f, -1 / 3f, -1 / 3f),
          Vector4(-1f, -1f, -1 / 3f, -1f)
        )), subfacesString
    )

  it should "contain top middle subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        flatSubfaces, List(
          Vector4(-1f, -1f, -1f, -1 / 3f),
          Vector4(-1f, -1f, -1f, 1 / 3f),
          Vector4(-1f, -1f, -1 / 3f, 1 / 3f),
          Vector4(-1f, -1f, -1 / 3f, -1 / 3f)
        )), subfacesString
    )

  it should "contain top right subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        flatSubfaces, List(
          Vector4(-1f, -1f, -1f, 1 / 3f),
          Vector4(-1f, -1f, -1f, 1f),
          Vector4(-1f, -1f, -1 / 3f, 1f),
          Vector4(-1f, -1f, -1 / 3f, 1 / 3f)
        )), subfacesString
    )

  it should "contain middle left subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        flatSubfaces, List(
          Vector4(-1f, -1f, -1 / 3f, -1f),
          Vector4(-1f, -1f, -1 / 3f, -1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, -1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, -1f)
        )), subfacesString
    )

  it should "not contain center subface" in new Sponge2:
    assert(!containsAllEpsilon(flatSubfaces, centerSubface), subfacesString)

  it should "contain middle right subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        flatSubfaces, List(
          Vector4(-1f, -1f, -1 / 3f, 1 / 3f),
          Vector4(-1f, -1f, -1 / 3f, 1f),
          Vector4(-1f, -1f, 1 / 3f, 1f),
          Vector4(-1f, -1f, 1 / 3f, 1 / 3f)
        )), subfacesString
    )

  it should "contain bottom left subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        flatSubfaces, List(
          Vector4(-1f, -1f, 1 / 3f, -1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, -1f),
          Vector4(-1f, -1f, 1f, -1f),
          Vector4(-1f, -1f, 1f, -1 / 3f)
        )), subfacesString
    )

  it should "contain bottom middle subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        flatSubfaces, List(
          Vector4(-1f, -1f, 1 / 3f, 1 / 3f),
          Vector4(-1f, -1f, 1f, 1 / 3f),
          Vector4(-1f, -1f, 1f, -1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, -1 / 3f)
        )), subfacesString
    )

  it should "contain bottom right subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        flatSubfaces, List(
          Vector4(-1f, -1f, 1 / 3f, 1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, 1f),
          Vector4(-1f, -1f, 1f, 1f),
          Vector4(-1f, -1f, 1f, 1 / 3f)
        )), subfacesString
    )

  it should "all have 1/9 the original area" in new Sponge2:
    val originalArea: Float = face.area
    val subfaceArea: Seq[Float] = flatSubfaces.map(_.area)
    subfaceArea.foreach(_ shouldBe originalArea / 9 +- epsilon)

  "A subdivided face's perpendicular parts" should "have size 8" in new Sponge2:
    perpendicularSubfaces should have size 8

  it should "have 8 distinct surfaces" in new Sponge2:
    perpendicularSubfaces.toSet should have size 8

  it should "not contain any of the flat surfaces" in new Sponge2:
    flatSubfaces.toSet.intersect(perpendicularSubfaces.toSet) shouldBe empty

  it should "all have the same base lines" in new Sponge2:
    perpendicularSubfaces.foreach(subface =>
      centerSubfaceEdges.exists(line => lineRoughlyEquals(line, (subface.a, subface.b))) shouldBe true
    )

  it should "all be 1/3 of the original side length long" in new Sponge2:
    perpendicularSubfaces.foreach(subface =>
      (subface.b - subface.a).len shouldBe 2/3f +- epsilon
    )

  it should "all be parallel to the axes" in new Sponge2:
    def isParallelToAxes(v: Vector4): Boolean =
      v.toArray.count(f => math.abs(f) < epsilon) == 3

    perpendicularSubfaces.foreach(r =>
      val differences = Seq(r.b - r.a, r.c - r.b, r.d - r.c, r.a - r.d)
      assert(
        differences.forall(v => isParallelToAxes(v)),
        differences.map(vec2string).toString
      )
    )

  it should "all have 1/9 the original area" in new Sponge2:
    val originalArea: Float = face.area
    val subfaceArea: Seq[Float] = perpendicularSubfaces.map(_.area)
    subfaceArea.foreach(_ shouldBe originalArea / 9 +- epsilon)

  it should "contain face rotated into y direction" in new Sponge2:
    private val expected = List(
      Vector4(-1f,   -1f, -1/3f, -1/3f),
      Vector4(-1f, -1/3f, -1/3f, -1/3f),
      Vector4(-1f, -1/3f, -1/3f,  1/3f),
      Vector4(-1f,   -1f, -1/3f,  1/3f)
    )
    assert(
      containsAllEpsilon(perpendicularSubfaces, expected),
      s"""\nexpected:
  ${faceToString(expected)}
actual:
 $perpendicularSubfacesString
diff: ${diffToFaces(perpendicularSubfaces, expected)}\n"""  //subfacesString
    )

  "A subdivided face" should "contain top left subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f, -1f, -1f),
          Vector4(-1f, -1f, -1f, -1 / 3f),
          Vector4(-1f, -1f, -1 / 3f, -1 / 3f),
          Vector4(-1f, -1f, -1 / 3f, -1f)
        )), subfacesString
    )

  it should "contain top middle subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f, -1f, -1 / 3f),
          Vector4(-1f, -1f, -1f, 1 / 3f),
          Vector4(-1f, -1f, -1 / 3f, 1 / 3f),
          Vector4(-1f, -1f, -1 / 3f, -1 / 3f)
        )), subfacesString
    )

  it should "contain top right subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f, -1f, 1 / 3f),
          Vector4(-1f, -1f, -1f, 1f),
          Vector4(-1f, -1f, -1 / 3f, 1f),
          Vector4(-1f, -1f, -1 / 3f, 1 / 3f)
        )), subfacesString
    )

  it should "contain middle left subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f, -1 / 3f, -1f),
          Vector4(-1f, -1f, -1 / 3f, -1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, -1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, -1f)
        )), subfacesString
    )

  it should "not contain center subface" in new Sponge2:
    assert(
      !containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f, -1 / 3f, 1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, 1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, -1 / 3f),
          Vector4(-1f, -1f, -1 / 3f, -1 / 3f)
        )), subfacesString
    )

  it should "contain middle right subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f, -1 / 3f, 1 / 3f),
          Vector4(-1f, -1f, -1 / 3f, 1f),
          Vector4(-1f, -1f, 1 / 3f, 1f),
          Vector4(-1f, -1f, 1 / 3f, 1 / 3f)
        )), subfacesString
    )

  it should "contain bottom left subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f, 1 / 3f, -1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, -1f),
          Vector4(-1f, -1f, 1f, -1f),
          Vector4(-1f, -1f, 1f, -1 / 3f)
        )), subfacesString
    )

  it should "contain bottom middle subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f, 1 / 3f, 1 / 3f),
          Vector4(-1f, -1f, 1f, 1 / 3f),
          Vector4(-1f, -1f, 1f, -1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, -1 / 3f)
        )), subfacesString
    )

  it should "contain bottom right subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f, 1 / 3f, 1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, 1f),
          Vector4(-1f, -1f, 1f, 1f),
          Vector4(-1f, -1f, 1f, 1 / 3f)
        )), subfacesString
    )

  ignore should "contain face pointing in x bordered on center hole" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f,   -1f, -1/3f, -1/3f),
          Vector4(-1f, -1/3f, -1/3f,  1/3f),
          Vector4(-1f, -1/3f,  1/3f,  1/3f),
          Vector4(-1f,   -1f,  1/3f, -1/3f)
        ), 1e-5), subfacesString
    )

  it should "contain 16 subfaces" in new Sponge2:
    assert(subfaces.size == 16)

  it should "contain 16 distinct subfaces" in new Sponge2:
    assert(subfaces.toSet.size == 16)

  private def round(f: Float): Float = math.round(f / epsilon) * epsilon

  "All level 1 sponge corner points" should "have absolute coordinate values 1/2 or 1/6" in new Sponge2:
    val cornerPoints: Seq[Vector4] = sponge2.faces.flatMap(_.asSeq)
    val cornerCoordinateValues: Set[Float] = cornerPoints.toSet.flatMap(_.toArray).map(math.abs).map(round)
    val clue: String = sponge2str(sponge2).replace("0.83", s"${Console.YELLOW}0.83${Console.RED}")
    withClue(clue) {cornerCoordinateValues should contain only (round(1/2f), round(1/6f))}

  for i <- 0 to 23 do
    s"Subdivided subface ${i+1}" should "have absolute coordinate values 1/2 or 1/6" in new Sponge2:
      checkCoordinatesOfSubdividedFace(TesseractSponge2(0), i)

  private def checkCoordinatesOfSubdividedFace(sponge: TesseractSponge2, i: Int) =
    val subface: Face4D = sponge.faces(i)
    val subdividedFace: Seq[Face4D] = sponge.subdividedFace(subface)
    val cornerPoints: Seq[Vector4] = subdividedFace.flatMap(_.asSeq)
    val cornerCoordinateValues: Set[Float] = cornerPoints.toSet.flatMap(_.toArray).map(math.abs).map(round)
    val coords = seq2str(subdividedFace).replace("0.83", s"${Console.YELLOW}0.83${Console.RED}")
    val clue = s"${i+1}/${sponge.faces.length} (${Plane(subface)})\n$coords\n"
    withClue(clue) {cornerCoordinateValues should contain atLeastOneOf(round(1 / 2f), round(1 / 6f))}
    withClue(clue) {cornerCoordinateValues diff Set(round(1 / 2f), round(1 / 6f)) shouldBe empty}

  private def containsEpsilon(vecs: Iterable[Vector4], vec: Vector4, epsilon: Float = 1e-6f): Boolean =
    vecs.exists(_.epsilonEquals(vec, epsilon))

  private def assertContainsEpsilon(vecs: Iterable[Vector4], vec: Vector4, epsilon: Float = 1e-6f): Unit =
    assert(containsEpsilon(vecs, vec, epsilon), vecs.toString)

  private def containsAllEpsilon(rects: Seq[Face4D], vecs: Seq[Vector4], epsilon: Float = 1e-6f): Boolean =
    containsAllEpsilon2(rects.map(_.asSeq), vecs, epsilon)

  private def containsAllEpsilon2(rects: Seq[Seq[Vector4]], vecs: Seq[Vector4], epsilon: Float = 1e-6f): Boolean =
    rects.exists(rect => rect.forall(v => containsEpsilon(vecs, v, epsilon)))
