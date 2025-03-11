package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PlaneSuite extends AnyFlatSpec with StandardVector with Matchers:
  "A plane" should "be creatable from an array of points in xy plane" in:
    val cornerPoints = Seq(zero, y, x+y, x)
    Plane(cornerPoints) should be (Plane.xy)

  it should "be creatable from an array of points in xz plane" in:
    val cornerPoints = Seq(zero, z, x + z, x)
    Plane(cornerPoints) should be (Plane.xz)

  it should "be creatable from an array of points in xw plane" in:
    val cornerPoints = Seq(zero, w, x + w, x)
    Plane(cornerPoints) should be (Plane.xw)

  it should "be creatable from an array of points in yz plane" in:
    val cornerPoints = Seq(zero, y, y + z, z)
    Plane(cornerPoints) should be (Plane.yz)

  it should "be creatable from an array of points in yw plane" in:
    val cornerPoints = Seq(zero, w, y + w, y)
    Plane(cornerPoints) should be (Plane.yw)

  it should "be creatable from an array of points in zw plane" in:
    val cornerPoints = Seq(zero, w, z + w, z)
    Plane(cornerPoints) should be (Plane.zw)

  it should "be equal for planes created from the same points" in:
    val points1 = Seq(zero, y, x + y, x)
    val points2 = Seq(zero, y, x + y, x)
    Plane(points1) should be (Plane(points2))

  it should "not be equal for planes created from different points" in:
    val points1 = Seq(zero, y, x + y, x)
    val points2 = Seq(zero, z, x + z, x)
    Plane(points1) should not be Plane(points2)

  it should "throw an exception for an empty array of points" in:
    val emptyPoints = Seq[Vector4]()
    an[IllegalArgumentException] should be thrownBy Plane(emptyPoints)

  it should "throw an exception for a single point" in:
    val singlePoint = Seq(zero)
    an[IllegalArgumentException] should be thrownBy Plane(singlePoint)

  it should "throw an exception for two points" in:
    val twoPoints = Seq(zero, x)
    an[IllegalArgumentException] should be thrownBy Plane(twoPoints)

  it should "throw an exception for duplicate points" in:
    val duplicatePoints = Seq(zero, x, x, y)
    an[IllegalArgumentException] should be thrownBy Plane(duplicatePoints)

  it should "throw an exception for non-planar points" in:
    val nonPlanarPoints = Seq(zero, x, y, z)
    an[IllegalArgumentException] should be thrownBy Plane(nonPlanarPoints)

  it should "throw an exception for collinear points" in:
    val collinearPoints = Seq(zero, x, x * 2, x * 3)
    an[IllegalArgumentException] should be thrownBy Plane(collinearPoints)

  it should "throw an exception for invalid indices" in:
    an[IllegalArgumentException] should be thrownBy Plane(-1, 4)
    an[IllegalArgumentException] should be thrownBy Plane(4, 0)
    an[IllegalArgumentException] should be thrownBy Plane(0, 4)

  "A plane's basis directions" should "be correct when created in the xy plane" in:
    val corners = Seq(zero, y, x+y, x)
    Plane.setIndices(corners) should contain only (0, 1)

  it should "be correct when created in the xz plane" in:
    val corners = Seq(zero, z, x + z, x)
    Plane.setIndices(corners) should contain only (0, 2)

  it should "be correct when created in the xw plane" in:
    val corners = Seq(zero, w, x + w, x)
    Plane.setIndices(corners) should contain only (0, 3)

  it should "be correct when created in the yz plane" in:
    val corners = Seq(zero, y, y + z, z)
    Plane.setIndices(corners) should contain only (1, 2)

  it should "be correct when created in the yw plane" in:
    val corners = Seq(zero, w, y + w, y)
    Plane.setIndices(corners) should contain only (1, 3)

  it should "be correct when created in the zw plane" in:
    val corners = Seq(zero, w, z + w, z)
    Plane.setIndices(corners) should contain only (2, 3)

  "A plane's defining points" should "have correct differences between them in the xy plane" in:
    val corners = Seq(zero, y, x+y, x)
    assert(seqsOfVectorsEpsilonEquals(Plane.differenceVectors(corners), Seq(y, x, -y, -x)))

  it should "have correct differences between them in the xz plane" in:
    val corners = Seq(zero, z, x + z, x)
    assert(seqsOfVectorsEpsilonEquals(Plane.differenceVectors(corners), Seq(z, x, -z, -x)))

  it should "have correct differences between them in the xw plane" in:
    val corners = Seq(zero, w, x + w, x)
    assert(seqsOfVectorsEpsilonEquals(Plane.differenceVectors(corners), Seq(w, x, -w, -x)))

  it should "have correct differences between them in the yz plane" in:
    val corners = Seq(zero, y, y + z, z)
    assert(seqsOfVectorsEpsilonEquals(Plane.differenceVectors(corners), Seq(y, z, -y, -z)))

  it should "have correct differences between them in the yw plane" in:
    val corners = Seq(zero, w, y + w, y)
    assert(seqsOfVectorsEpsilonEquals(Plane.differenceVectors(corners), Seq(w, y, -w, -y)))

  it should "have correct differences between them in the zw plane" in:
    val corners = Seq(zero, w, z + w, z)
    assert(seqsOfVectorsEpsilonEquals(Plane.differenceVectors(corners), Seq(w, z, -w, -z)))

