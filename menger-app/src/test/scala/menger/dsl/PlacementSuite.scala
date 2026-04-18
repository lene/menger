package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PlacementSuite extends AnyFlatSpec with Matchers:

  private val proto = SceneNode.leaf(Sphere())
  private val eps   = 1e-4f

  private def approx(a: Float, b: Float): Boolean = math.abs(a - b) < eps

  private def positions(node: SceneNode): List[Vec3] =
    node.children.map(_.transform.translation)

  // --- grid ---

  "Placement.grid" should "produce rows × cols children" in:
    Placement.grid(2, 3, 1f)(proto).children should have size 6

  it should "center the grid at the origin" in:
    val pos = positions(Placement.grid(3, 3, 2f)(proto))
    pos.map(_.x).min should be < 0f
    pos.map(_.x).max should be > 0f
    pos.map(_.z).min should be < 0f
    pos.map(_.z).max should be > 0f

  it should "space children by the given spacing" in:
    val pos = positions(Placement.grid(1, 2, 3f)(proto)).sortBy(_.x)
    approx(pos(1).x - pos(0).x, 3f) shouldBe true

  it should "keep all Y coordinates at zero" in:
    positions(Placement.grid(3, 3, 1f)(proto)).foreach(_.y shouldBe 0f)

  it should "reject rows or cols <= 0" in:
    an[IllegalArgumentException] should be thrownBy Placement.grid(0, 3, 1f)(proto)
    an[IllegalArgumentException] should be thrownBy Placement.grid(3, 0, 1f)(proto)

  // --- ring ---

  "Placement.ring" should "produce the requested number of children" in:
    Placement.ring(6, 2f)(proto).children should have size 6

  it should "place all children at the given radius" in:
    positions(Placement.ring(8, 3f)(proto)).foreach { p =>
      val r = math.sqrt(p.x * p.x + p.z * p.z).toFloat
      approx(r, 3f) shouldBe true
    }

  it should "keep all Y coordinates at zero" in:
    positions(Placement.ring(4, 1f)(proto)).foreach(_.y shouldBe 0f)

  it should "distribute copies evenly (equal arc spacing)" in:
    val pos   = positions(Placement.ring(4, 1f)(proto))
    val angles = pos.map(p => math.atan2(p.z, p.x).toFloat).sorted
    val diffs  = angles.zip(angles.tail).map { case (a, b) => b - a }
    diffs.foreach(d => approx(d, (math.Pi / 2).toFloat) shouldBe true)

  it should "reject count <= 0" in:
    an[IllegalArgumentException] should be thrownBy Placement.ring(0, 1f)(proto)

  // --- spiral ---

  "Placement.spiral" should "produce the requested number of children" in:
    Placement.spiral(5, 1f, 3f, 1f)(proto).children should have size 5

  it should "place the first child at radiusStart on the positive X axis" in:
    val first = positions(Placement.spiral(3, 2f, 4f, 1f)(proto)).head
    approx(first.x, 2f) shouldBe true
    approx(first.z, 0f) shouldBe true

  it should "place the last child at radiusEnd after the given turns" in:
    val last = positions(Placement.spiral(3, 1f, 3f, 1f)(proto)).last
    val r    = math.sqrt(last.x * last.x + last.z * last.z).toFloat
    approx(r, 3f) shouldBe true

  it should "keep all Y coordinates at zero" in:
    positions(Placement.spiral(4, 1f, 2f, 0.5f)(proto)).foreach(_.y shouldBe 0f)

  it should "reject count <= 0" in:
    an[IllegalArgumentException] should be thrownBy Placement.spiral(0, 1f, 2f, 1f)(proto)

  // --- scatter ---

  "Placement.scatter" should "produce the requested number of children" in:
    Placement.scatter(7, Vec3(1f, 1f, 1f))(proto).children should have size 7

  it should "keep all positions within the given bounds" in:
    val bounds = Vec3(2f, 1f, 3f)
    positions(Placement.scatter(20, bounds)(proto)).foreach { p =>
      math.abs(p.x) should be <= bounds.x
      math.abs(p.y) should be <= bounds.y
      math.abs(p.z) should be <= bounds.z
    }

  it should "be deterministic for the same seed" in:
    val a = positions(Placement.scatter(5, Vec3(1f, 1f, 1f), seed = 7L)(proto))
    val b = positions(Placement.scatter(5, Vec3(1f, 1f, 1f), seed = 7L)(proto))
    a shouldBe b

  it should "differ for different seeds" in:
    val a = positions(Placement.scatter(5, Vec3(1f, 1f, 1f), seed = 1L)(proto))
    val b = positions(Placement.scatter(5, Vec3(1f, 1f, 1f), seed = 2L)(proto))
    a should not equal b

  it should "reject count <= 0" in:
    an[IllegalArgumentException] should be thrownBy
      Placement.scatter(0, Vec3(1f, 1f, 1f))(proto)

  // --- prototype preservation ---

  "Placement helpers" should "preserve prototype's own transform as child offset" in:
    val shifted = SceneNode.leaf(Transform.at(Vec3(0f, 5f, 0f)), Sphere())
    val node    = Placement.ring(1, 2f)(shifted)
    // The ring group child has translation (2,0,0); its child is `shifted` with y=5
    node.children.head.children.head.transform.translation shouldBe Vec3(0f, 5f, 0f)
