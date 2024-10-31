package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec

class TesseractSponge2Suite extends AnyFlatSpec with RectMesh:
  trait Sponge2:
    val tesseract: Tesseract = Tesseract(2)
    val sponge2: TesseractSponge2 = TesseractSponge2(1)
    val face: RectVertices4D = tesseract.faces.head
    assert(face == (
      Vector4(-1, -1, -1, -1), Vector4(-1, -1, -1, 1), Vector4(-1, -1, 1, 1), Vector4(-1, -1, 1, -1))
    )
    val subfaces: Seq[RectVertices4D] = sponge2.subdividedFace(face)
    val subfacesString: String = subfaces.toString.replace("),", "),\n")
    val flatSubfaces: Seq[RectVertices4D] = sponge2.subdivideFlatParts(face)
    val perpendicularSubfaces: Seq[RectVertices4D] = sponge2.subdividePerpendicularParts(face)
    val perpendicularSubfacesString: String = perpendicularSubfaces.toString.replace("),", "),\n")

  "A TesseractSponge2 level 0" should "have 24 faces" in:
    val sponge = TesseractSponge2(0)
    assert(sponge.faces.size == 24)

  "A TesseractSponge level < 0" should "be imposssible" in:
    assertThrows[AssertionError] {
      TesseractSponge2(-1)
    }

  "A subdivided face's corner points" should "contain the original face's corners" in new Sponge2:
    for v <- face.toList do assertContainsEpsilon(sponge2.cornerPoints(face).values, v)

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
    assert(
      !containsAllEpsilon(
        flatSubfaces, List(
          Vector4(-1f, -1f, -1 / 3f, 1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, 1 / 3f),
          Vector4(-1f, -1f, 1 / 3f, -1 / 3f),
          Vector4(-1f, -1f, -1 / 3f, -1 / 3f)
        )), subfacesString
    )

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

  "A subdivided face's perpendicular parts" should "have size 8" in new Sponge2:
    assert(perpendicularSubfaces.size == 8)

  it should "have 8 distinct surfaces" in new Sponge2:
    assert(perpendicularSubfaces.toSet.size == 8)

  it should "not contain any of the flat surfaces" in new Sponge2:
    assert(
      flatSubfaces.toSet.intersect(perpendicularSubfaces.toSet).isEmpty
    )

  it should "contain face rotated into y direction" in new Sponge2:
    assert(
      containsAllEpsilon(
        perpendicularSubfaces, List(
          Vector4(-1f,   -1f, -1/3f, -1/3f),
          Vector4(-1f, -1/3f, -1/3f, -1/3f),
          Vector4(-1f, -1/3f, -1/3f,  1/3f),
          Vector4(-1f,   -1f, -1/3f,  1/3f)
        )), subfacesString
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

  ignore should "not contain center subface" in new Sponge2:
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
        )), subfacesString
    )

  it should "contain 16 subfaces" in new Sponge2:
    assert(subfaces.size == 16)

  ignore should "contain 16 distinct subfaces" in new Sponge2:
    assert(subfaces.toSet.size == 16)
  
  def containsEpsilon(vecs: Iterable[Vector4], vec: Vector4, epsilon: Float = 1e-6f): Boolean =
    vecs.exists(_.epsilonEquals(vec, epsilon))

  def assertContainsEpsilon(vecs: Iterable[Vector4], vec: Vector4, epsilon: Float = 1e-6f): Unit =
    assert(containsEpsilon(vecs, vec, epsilon), vecs.toString)

  def containsAllEpsilon(rects: Seq[RectVertices4D], vecs: Seq[Vector4], epsilon: Float = 1e-6f): Boolean =
    containsAllEpsilon2(rects.map(_.toList.map(_.asInstanceOf[Vector4])), vecs, epsilon)

  def containsAllEpsilon2(rects: Seq[Seq[Vector4]], vecs: Seq[Vector4], epsilon: Float = 1e-6f): Boolean =
    rects.exists(rect => rect.forall(v => containsEpsilon(vecs, v, epsilon)))
