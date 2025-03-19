package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.*
import CustomMatchers.*
import com.typesafe.scalalogging.LazyLogging
import org.scalatest.Inspectors.forAll

trait StandardVector:
  val axisName = Array("x", "y", "z", "w")
  val zero = Vector4(0, 0, 0, 0)
  val x = Vector4(1, 0, 0, 0)
  val y = Vector4(0, 1, 0, 0)
  val z = Vector4(0, 0, 1, 0)
  val w = Vector4(0, 0, 0, 1)

class RotationSuite extends AnyFlatSpec with Matchers with LazyLogging with StandardVector:
  "zero Rotation" should "keep point same" in:
    val p = Vector4(1, 2, 3, 4)
    val r = Rotation(0, 0, 0)
    r(p) should be (p)
  
  it should "keep Tesseract same" in:
    val t = Tesseract(1)
    val r = Rotation(0, 0, 0)
    r(t.vertices) should be (t.vertices)

  "value outside [0, 360)" should "be mapped to 0 if equals 360" in:
    Rotation(360, 0, 0).isZero should be (true)

  it should "be mapped to 0 if multiples of 360°" in:
    Seq(0f, 360f, 720f, -1080f).foreach { angle =>
      Rotation(angle, 0, 0).isZero should be (true)
      Rotation(0, angle, 0).isZero should be (true)
      Rotation(angle, angle, angle).isZero should be (true)
    }

  it should "be positive if > 360°" in:
    Rotation(370, 0, 0) === Rotation(10, 0, 0) should be (true)

  it should "be positive if < 0°" in:
    Rotation(-10, 0, 0) === Rotation(350, 0, 0) should be (true)

  "adding two rotations" should "be positive if sum < 360°" in:
    val r1 = Rotation(10, 0, 0)
    val r2 = Rotation(20, 0, 0)
    (r1 + r2) === Rotation(30, 0, 0) should be (true)

  it should "be positive if sum > 360°" in:
    val r1 = Rotation(350, 0, 0)
    val r2 = Rotation(20, 0, 0)
    (r1 + r2) === Rotation(10, 0, 0) should be (true)

  it should "be zero if sum == 360°" in:
    val r1 = Rotation(350, 0, 0)
    val r2 = Rotation(10, 0, 0)
    (r1 + r2).isZero should be (true)

  it should "add up" in:
    val r1 = Rotation(10, 0, 0)
    val r2 = Rotation(20, 0, 0)
    (r1 + r2) === Rotation(30, 0, 0) should be (true)

  "chaining two rotations" should "have same result regardless of order" in:
    val r1 = Rotation(10, 0, 0)
    val r2 = Rotation(20, 0, 0)
    val p = Vector4(1, 2, 3, 4)
    r1(r2(p)) === r2(r1(p)) should be (true)

  "rotating 90 degrees" should "work around xw plane" in:
    val r = Rotation(90, 0, 0)
    r(Vector4(1, 0, 0, 0)) should epsilonEqual (Vector4(0, 0, 0, -1))

  it should "work around yw plane" in:
    val r = Rotation(0, 90, 0)
    r(Vector4(0, 1, 0, 0)) should epsilonEqual (Vector4(0, 0, 0, -1))

  it should "work around zw plane" in:
    val r = Rotation(0, 0, 90)
    r(Vector4(0, 0, 1, 0)) should epsilonEqual (Vector4(0, 0, 0, -1))

  "printing all 4D base transformation matrices" should "be possible" in:
    val axisNames = Seq("x", "y", "z", "w")
    for i <- 0 to 3 do
      for j <- 0 to 3 do
        if i < j then
          val rotate = Rotation(Plane(i, j), 90, Vector4.Zero)
          logger.debug(s"Rotating around ${axisNames(i)}${axisNames(j)} plane:")
          logger.debug(s"\n${rotate.transformationMatrix.str}")

  // TODO: make this return a function which takes expected as argument
  private def checkPlaneRotation(
    point: Vector4, plane: Plane, expected: Vector4, angle: Float = 90
  ): Unit =
    val rotate = Rotation(plane, angle, Vector4.Zero)
    val rotated = rotate(point)
    rotated should epsilonEqual(expected)


  // TODO: make this return a function which takes expected1 and expected2 as arguments
  private def checkPlaneAndLineRotation(
    point: Vector4, plane: Plane, direction: Edge, expected1: Vector4, expected2: Vector4
  ): Unit =
    val rotate = Rotation(plane, direction, direction(0), 90)
    rotate should have length 2
    val rotated = rotate.map(_.apply(point))
    rotated(0) should epsilonEqual(expected1)
    rotated(1) should epsilonEqual(expected2)

  "rotating around a plane and an axis" should "be correct for rotating around x in the xy plane" in :
    checkPlaneAndLineRotation(y, Plane.xy, Edge(zero, x), -z, -w)

  it should "be correct for rotating around y in the xy plane" in :
    checkPlaneAndLineRotation(x, Plane.xy, Edge(zero, y), z, w)

  it should "fail for rotating around z in the xy plane" in :
    an [IllegalArgumentException] should be thrownBy Rotation(Plane.xy, Edge(zero, z), zero, 90)

  it should "fail for rotating around w in the xy plane" in :
    an [IllegalArgumentException] should be thrownBy Rotation(Plane.xy, Edge(zero, w), zero, 90)

  it should "be correct for rotating around x in the xz plane" in :
    checkPlaneAndLineRotation(z, Plane.xz, Edge(zero, x), -y, -w)

  it should "be correct for rotating around z in the xz plane" in :
    checkPlaneAndLineRotation(x, Plane.xz, Edge(zero, z), y, w)

  it should "fail for rotating around y in the xz plane" in :
    an [IllegalArgumentException] should be thrownBy Rotation(Plane.xz, Edge(zero, y), zero, 90)

  it should "fail for rotating around w in the xz plane" in :
    an [IllegalArgumentException] should be thrownBy Rotation(Plane.xz, Edge(zero, w), zero, 90)

  it should "be correct for rotating around x in the xw plane" in :
    checkPlaneAndLineRotation(w, Plane.xw, Edge(zero, x), -y, -z)

  it should "be correct for rotating around w in the xw plane" in :
    checkPlaneAndLineRotation(x, Plane.xw, Edge(zero, w), y, z)

  it should "fail for rotating around y in the xw plane" in :
    an [IllegalArgumentException] should be thrownBy Rotation(Plane.xw, Edge(zero, y), zero, 90)

  it should "fail for rotating around z in the xw plane" in :
    an [IllegalArgumentException] should be thrownBy Rotation(Plane.xw, Edge(zero, z), zero, 90)

  it should "be correct for rotating around y in the yz plane" in :
    checkPlaneAndLineRotation(z, Plane.yz, Edge(zero, y), -x, -w)

  it should "be correct for rotating around z in the yz plane" in :
    checkPlaneAndLineRotation(y, Plane.yz, Edge(zero, z), x, w)

  it should "fail for rotating around x in the yz plane" in :
    an [IllegalArgumentException] should be thrownBy Rotation(Plane.yz, Edge(zero, x), zero, 90)

  it should "fail for rotating around w in the yz plane" in :
    an [IllegalArgumentException] should be thrownBy Rotation(Plane.yz, Edge(zero, w), zero, 90)

  it should "be correct for rotating around y in the yw plane" in :
    checkPlaneAndLineRotation(w, Plane.yw, Edge(zero, y), -x, -z)

  it should "be correct for rotating around w in the yw plane" in :
    checkPlaneAndLineRotation(y, Plane.yw, Edge(zero, w), x, z)

  it should "fail for rotating around x in the yw plane" in :
    an [IllegalArgumentException] should be thrownBy Rotation(Plane.yw, Edge(zero, x), zero, 90)

  it should "fail for rotating around z in the yw plane" in :
    an [IllegalArgumentException] should be thrownBy Rotation(Plane.yw, Edge(zero, z), zero, 90)

  it should "be correct for rotating around z in the zw plane" in :
    checkPlaneAndLineRotation(w, Plane.zw, Edge(zero, z), -x, -y)

  it should "be correct for rotating around w in the zw plane" in :
    checkPlaneAndLineRotation(z, Plane.zw, Edge(zero, w), x, y)

  it should "fail for rotating around x in the zw plane" in :
    an [IllegalArgumentException] should be thrownBy Rotation(Plane.zw, Edge(zero, x), zero, 90)

  it should "fail for rotating around y in the zw plane" in :
    an [IllegalArgumentException] should be thrownBy Rotation(Plane.zw, Edge(zero, y), zero, 90)

  "rotating a point around different planes" should "be correct for xy plane" in :
    checkPlaneRotation(x, Plane.xy, -y)
    checkPlaneRotation(y, Plane.xy, x)
    checkPlaneRotation(z, Plane.xy, z)
    checkPlaneRotation(w, Plane.xy, w)

  it should "be correct for xz plane" in :
    checkPlaneRotation(x, Plane.xz, -z)
    checkPlaneRotation(y, Plane.xz, y)
    checkPlaneRotation(z, Plane.xz, x)
    checkPlaneRotation(w, Plane.xz, w)

  it should "be correct for xw plane" in :
    checkPlaneRotation(x, Plane.xw, -w)
    checkPlaneRotation(y, Plane.xw, y)
    checkPlaneRotation(z, Plane.xw, z)
    checkPlaneRotation(w, Plane.xw, x)

  it should "be correct for yz plane" in :
    checkPlaneRotation(x, Plane.yz, x)
    checkPlaneRotation(y, Plane.yz, -z)
    checkPlaneRotation(z, Plane.yz, y)
    checkPlaneRotation(w, Plane.yz, w)

  it should "be correct for yw plane" in :
    checkPlaneRotation(x, Plane.yw, x)
    checkPlaneRotation(y, Plane.yw, -w)
    checkPlaneRotation(z, Plane.yw, z)
    checkPlaneRotation(w, Plane.yw, y)

  it should "be correct for zw plane" in :
    checkPlaneRotation(x, Plane.zw, x)
    checkPlaneRotation(y, Plane.zw, y)
    checkPlaneRotation(z, Plane.zw, -w)
    checkPlaneRotation(w, Plane.zw, z)

  "180 degree rotation" should "flip a vector if it lies in the rotation plane" in :
    checkPlaneRotation(x, Plane.xy, -x, 180)
    checkPlaneRotation(y, Plane.xy, -y, 180)
    checkPlaneRotation(x, Plane.xz, -x, 180)
    checkPlaneRotation(z, Plane.xz, -z, 180)
    checkPlaneRotation(x, Plane.xw, -x, 180)
    checkPlaneRotation(w, Plane.xw, -w, 180)
    checkPlaneRotation(y, Plane.yz, -y, 180)
    checkPlaneRotation(z, Plane.yz, -z, 180)
    checkPlaneRotation(y, Plane.yw, -y, 180)
    checkPlaneRotation(w, Plane.yw, -w, 180)
    checkPlaneRotation(z, Plane.zw, -z, 180)
    checkPlaneRotation(w, Plane.zw, -w, 180)

  it should "leave a vector intact if not in the rotation plane" in :
    checkPlaneRotation(z, Plane.xy, z, 180)
    checkPlaneRotation(w, Plane.xy, w, 180)
    checkPlaneRotation(y, Plane.xz, y, 180)
    checkPlaneRotation(w, Plane.xz, w, 180)
    checkPlaneRotation(y, Plane.xw, y, 180)
    checkPlaneRotation(z, Plane.xw, z, 180)
    checkPlaneRotation(x, Plane.yz, x, 180)
    checkPlaneRotation(w, Plane.yz, w, 180)
    checkPlaneRotation(x, Plane.yw, x, 180)
    checkPlaneRotation(z, Plane.yw, z, 180)
    checkPlaneRotation(x, Plane.zw, x, 180)
    checkPlaneRotation(y, Plane.zw, y, 180)

  "Adding a 90 degree rotation to a 90 degree rotation" should "be a 180 degree rotation" in :
    val rotate1 = Rotation(Plane.xy, 90)
    val rotate2 = Rotation(Plane.xy, 90)
    val rotated = rotate2(rotate1(x))
    rotated should epsilonEqual (-x)

  "rotating by negative angles" should "produce correct results" in :
    checkPlaneRotation(x, Plane.xy, y, -90)
    checkPlaneRotation(y, Plane.xy, -x, -90)
    checkPlaneRotation(z, Plane.xz, -x, -90)
    checkPlaneRotation(w, Plane.xw, -x, -90)

  "rotating by non-standard angles" should "produce correct results" in :
    val sqrt2_2 = math.sqrt(2).toFloat / 2
    checkPlaneRotation(x, Plane.xy, Vector4(sqrt2_2, -sqrt2_2, 0, 0), 45)
    checkPlaneRotation(x, Plane.xy, Vector4(-sqrt2_2, -sqrt2_2, 0, 0), 135)
    checkPlaneRotation(y, Plane.xy, Vector4(sqrt2_2, sqrt2_2, 0, 0), 45)
    checkPlaneRotation(z, Plane.xz, Vector4(sqrt2_2, 0, sqrt2_2, 0), 45)
    checkPlaneRotation(w, Plane.xw, Vector4(sqrt2_2, 0, 0, sqrt2_2), 45)

  "rotating around multiple planes" should "produce correct results" in :
    val rotateXZ = Rotation(Plane.xz, 90)
    val rotateYZ = Rotation(Plane.yz, 90)
    val rotatedXZYZ = rotateYZ(rotateXZ(x))
    rotatedXZYZ should epsilonEqual (-y)

  "A concrete example from the 4d sponge" should "produce correct results in xy plane" in :
    val centralPart = Seq(
      Vector4(-1, -1, 3, 3), Vector4(-1, 1, 3, 3),
      Vector4(1, 1, 3, 3), Vector4(1, -1, 3, 3)
    )
    checkFacesAroundFaceEdges(centralPart, Plane.xy, -90f)

  it should "produce correct results in xy plane pointing opposite" in :
    val centralPart = Seq(
      Vector4(-1, -1, -3, -3), Vector4(-1, 1, -3, -3),
      Vector4(1, 1, -3, -3), Vector4(1, -1, -3, -3)
    )
    checkFacesAroundFaceEdges(centralPart, Plane.xy, 90f)

  it should "produce correct results in xz plane" in :
    val centralPart = Seq(
      Vector4(-1, 3, -1, 3), Vector4(-1, 3, 1, 3),
      Vector4(1, 3, 1, 3), Vector4(1, 3, -1, 3)
    )
    checkFacesAroundFaceEdges(centralPart, Plane.xz, -90f)

  it should "produce correct results in xz plane pointing opposite" in :
    val centralPart = Seq(
      Vector4(-1, -3, -1, -3), Vector4(-1, -3, 1, -3),
      Vector4(1, -3, 1, -3), Vector4(1, -3, -1, -3)
    )
    checkFacesAroundFaceEdges(centralPart, Plane.xz, 90f)

  it should "produce correct results in xw plane" in :
    val centralPart = Seq(
      Vector4(-1, 3, 3, -1), Vector4(-1, 3, 3, 1),
      Vector4(1, 3, 3, 1), Vector4(1, 3, 3, -1)
    )
    checkFacesAroundFaceEdges(centralPart, Plane.xw, -90f)

  it should "produce correct results in xw plane pointing opposite" in :
    val centralPart = Seq(
      Vector4(-1, -3, -3, -1), Vector4(-1, -3, -3, 1),
      Vector4(1, -3, -3, 1), Vector4(1, -3, -3, -1)
    )
    checkFacesAroundFaceEdges(centralPart, Plane.xw, 90f)

  it should "produce correct results in yz plane" in :
    val centralPart = Seq(
      Vector4(3, -1, -1, 3), Vector4(3, -1, 1, 3),
      Vector4(3, 1, 1, 3), Vector4(3, 1, -1, 3)
    )
    checkFacesAroundFaceEdges(centralPart, Plane.yz, -90f)

  it should "produce correct results in yz plane pointing opposite" in :
    val centralPart = Seq(
      Vector4(-3, -1, -1, -3), Vector4(-3, -1, 1, -3),
      Vector4(-3, 1, 1, -3), Vector4(-3, 1, -1, -3)
    )
    checkFacesAroundFaceEdges(centralPart, Plane.yz, 90f)

  it should "produce correct results in yw plane" in :
    val centralPart = Seq(
      Vector4(-3, -3, -1, -1), Vector4(-3, -1, -1, -1),
      Vector4(-3, -1, -1, 1), Vector4(-3, -3, -1, 1)
    )
    checkFacesAroundFaceEdges(centralPart, Plane.yw, -90f)

  it should "produce correct results in yw plane pointing opposite" in :
    val centralPart = Seq(
      Vector4(-3, -3, 1, 1), Vector4(-3, -1, 1, 1),
      Vector4(-3, -1, 1, -1), Vector4(-3, -3, 1, -1)
    )
    checkFacesAroundFaceEdges(centralPart, Plane.yw, 90f)

  it should "produce correct results in zw plane" in :
    val centralPart = Seq(
      Vector4(3, 3, -1, -1), Vector4(3, 3, -1, 1),
      Vector4(3, 3, 1, 1), Vector4(3, 3, 1, -1)
    )
    checkFacesAroundFaceEdges(centralPart, Plane.zw, -90f)

  it should "produce correct results in zw plane pointing opposite" in :
    val centralPart = Seq(
      Vector4(-3, -3, -1, -1), Vector4(-3, -3, -1, 1),
      Vector4(-3, -3, 1, 1), Vector4(-3, -3, 1, -1)
    )
    checkFacesAroundFaceEdges(centralPart, Plane.zw, 90f)

  def checkFacesAroundFaceEdges(centralPart: Seq[Vector4], rotationPlane: Plane, angle: Float): Unit =
    val edges = Seq(
      Edge(centralPart(0), centralPart(1)), Edge(centralPart(1), centralPart(2)),
      Edge(centralPart(2), centralPart(3)), Edge(centralPart(3), centralPart(0))
    )
    val oppositeEdges = edges.drop(2) ++ edges.take(2)
    (0 to 1).foreach(rotationDirection => {
      edges.indices.foreach(edge => {
        val rotatedRect = Face4D(
          edges(edge)(0), edges(edge)(1),
          Rotation(rotationPlane, edges(edge), edges(edge)(0), angle)(rotationDirection)(oppositeEdges(edge)(0)),
          Rotation(rotationPlane, edges(edge), edges(edge)(1), angle)(rotationDirection)(oppositeEdges(edge)(1))
        )

        val clue = s"rotated face:\n$rotatedRect\n"
        // all rotated square faces should have size 2x2
        withClue(clue) {rotatedRect.area shouldBe 4f +- 1e-6f}
        // all coordinates in this example should be +-1 or +-3
        withClue(clue) { forAll (rotatedRect.asSeq) { absElements(_) should contain atLeastOneOf (1, 3) } }
        withClue(clue) { forAll (rotatedRect.asSeq) { absElements(_).diff(Set(1, 3)) shouldBe empty } }
      })
    })

  def absElements(vec: Vector4): Set[Int] =
    vec.toArray.map(value => (math.abs(value) * 1e5).round.toInt / 100000).toSet
