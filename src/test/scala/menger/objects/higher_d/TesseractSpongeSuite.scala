package menger.objects.higher_d

import com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo
import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec

class TesseractSpongeSuite extends AnyFlatSpec with RectMesh:

  trait Sponge:
    val sponge: TesseractSponge = TesseractSponge(1)

  "A TesseractSponge level 1" should "have 48 times the number of a Tesseract's faces" in new Sponge:
    assert(sponge.faces.size == 48 * Tesseract().faces.size)

  it should "have no vertices with absolute value greater than 0.5" in new Sponge:
    assert(
      sponge.faces.forall(rect => rect.toList.forall(v => v.asInstanceOf[Vector4].toArray.forall(_.abs <= 0.5)))
    )

  it should "have no face in the center of each face of the Tesseract" in new Sponge:
    assert(sponge.faces.forall(rect => !isCenterOfOriginalFace(rect)))

  it should "have no face around the removed center Tesseract" in new Sponge:
    assert(sponge.faces.forall(rect => !isCenterOfOriginalTesseract(rect)))


  def isCenterOfOriginalFace(face: RectVertices4D): Boolean =
    // A face is a center face if 2 of its coordinates are +/- 1/6 and the other 2 are 0.5
    face.toList.forall({ v =>
      val va = v.asInstanceOf[Vector4].toArray
      va.count(_.abs == 0.5) == 2 && va.count(_.abs == 1 / 6f) == 2
    }
    )

  def isCenterOfOriginalTesseract(face: RectVertices4D): Boolean =
    // A face is a center face if all of its coordinates are +/- 1/6
    face.toList.forall { _.asInstanceOf[Vector4].toArray.count(_.abs == 1 / 6f) == 4 }

class TesseractSponge2Suite extends AnyFlatSpec with RectMesh:
  trait Sponge2:
    val tesseract: Tesseract = Tesseract(2)
    val sponge2: TesseractSponge2 = TesseractSponge2(1)
    val face: RectVertices4D = tesseract.faces.head
    assert(face == (
      Vector4(-1, -1, -1, -1), Vector4(-1, -1, -1, 1), Vector4(-1, -1, 1, 1), Vector4(-1, -1, 1, -1))
    )
    val subfaces: Seq[RectVertices4D] = sponge2.subdividedFace(face)

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

  "A subdivided face" should "contain top left subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f,   -1f,   -1f),
          Vector4(-1f, -1f,   -1f, -1/3f),
          Vector4(-1f, -1f, -1/3f, -1/3f),
          Vector4(-1f, -1f, -1/3f,   -1f)
      )), subfaces.toString
    )

  it should "contain top middle subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
        Vector4(-1f, -1f,   -1f, -1/3f),
        Vector4(-1f, -1f,   -1f,  1/3f),
        Vector4(-1f, -1f, -1/3f,  1/3f),
        Vector4(-1f, -1f, -1/3f, -1/3f)
      )), subfaces.toString
    )

  it should "contain top right subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f,   -1f,  1/3f),
          Vector4(-1f, -1f,   -1f,    1f),
          Vector4(-1f, -1f, -1/3f,    1f),
          Vector4(-1f, -1f, -1/3f,  1/3f)
        )), subfaces.toString
    )

  it should "contain middle left subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f, -1/3f,   -1f),
          Vector4(-1f, -1f, -1/3f, -1/3f),
          Vector4(-1f, -1f,  1/3f, -1/3f),
          Vector4(-1f, -1f,  1/3f,   -1f)
        )), subfaces.toString
    )

  it should "not contain center subface" in new Sponge2:
    assert(
      !containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f, -1/3f,  1/3f),
          Vector4(-1f, -1f,  1/3f,  1/3f),
          Vector4(-1f, -1f,  1/3f, -1/3f),
          Vector4(-1f, -1f, -1/3f, -1/3f)
        )), subfaces.toString
    )

  it should "contain middle right subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f, -1/3f,  1/3f),
          Vector4(-1f, -1f, -1/3f,    1f),
          Vector4(-1f, -1f,  1/3f,    1f),
          Vector4(-1f, -1f,  1/3f,  1/3f)
        )), subfaces.toString
    )

  it should "contain bottom left subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f,  1/3f, -1/3f),
          Vector4(-1f, -1f,  1/3f,   -1f),
          Vector4(-1f, -1f,    1f,   -1f),
          Vector4(-1f, -1f,    1f, -1/3f)
        )), subfaces.toString
    )

  it should "contain bottom middle subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f,  1/3f,  1/3f),
          Vector4(-1f, -1f,    1f,  1/3f),
          Vector4(-1f, -1f,    1f, -1/3f),
          Vector4(-1f, -1f,  1/3f, -1/3f)
        )), subfaces.toString
    )

  it should "contain bottom right subface" in new Sponge2:
    assert(
      containsAllEpsilon(
        subfaces, List(
          Vector4(-1f, -1f,  1/3f,  1/3f),
          Vector4(-1f, -1f,  1/3f,    1f),
          Vector4(-1f, -1f,    1f,    1f),
          Vector4(-1f, -1f,    1f,  1/3f)
        )), subfaces.toString
    )

  def containsEpsilon(vecs: Iterable[Vector4], vec: Vector4, epsilon: Float = 1e-6f): Boolean =
    vecs.exists(_.epsilonEquals(vec, epsilon))

  def assertContainsEpsilon(vecs: Iterable[Vector4], vec: Vector4, epsilon: Float = 1e-6f): Unit =
    assert(containsEpsilon(vecs, vec, epsilon), vecs.toString)

  def containsAllEpsilon(rects: Seq[RectVertices4D], vecs: Seq[Vector4], epsilon: Float = 1e-6f): Boolean =
    containsAllEpsilon2(rects.map(_.toList), vecs, epsilon)

  def containsAllEpsilon2(rects: Seq[Seq[Vector4]], vecs: Seq[Vector4], epsilon: Float = 1e-6f): Boolean =
    rects.exists(rect => rect.forall(v => containsEpsilon(vecs, v, epsilon)))
