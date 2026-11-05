package menger.engines.scene

import menger.cli.ObjectSpecParser
import menger.dsl.Cylinder
import menger.dsl.Vec3
import menger.engines.scene.AxialPrimitiveSceneBuilder.Axis
import menger.spec.CylinderGeometry
import menger.spec.ObjectRotation
import menger.spec.ObjectSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CylinderSceneBuilderTest extends AnyFlatSpec with Matchers:

  private val Tolerance = 1e-4f

  private def near(actual: (Float, Float, Float), expected: (Float, Float, Float)): Unit =
    actual._1 shouldBe expected._1 +- Tolerance
    actual._2 shouldBe expected._2 +- Tolerance
    actual._3 shouldBe expected._3 +- Tolerance

  private def axisNear(actual: Axis, expected: Axis): Unit =
    near(actual.start, expected.start)
    near(actual.end, expected.end)
    actual.radius shouldBe expected.radius +- Tolerance

  "CylinderSceneBuilder.validate" should "accept a valid cylinder spec" in:
    val spec = ObjectSpec(
      objectType = "cylinder",
      cylinder = CylinderGeometry(p0 = Some((0f, -0.5f, 0f)), p1 = Some((0f, 0.5f, 0f)))
    )
    CylinderSceneBuilder().validate(List(spec), 64) shouldBe Right(())

  it should "reject non-cylinder specs" in:
    CylinderSceneBuilder().validate(List(ObjectSpec(objectType = "cone")), 64) shouldBe
      a[Left[?, ?]]

  it should "reject an empty spec list" in:
    CylinderSceneBuilder().validate(List.empty, 64) shouldBe a[Left[?, ?]]

  it should "reject more specs than instances" in:
    val specs = List.fill(5)(ObjectSpec(objectType = "cylinder"))
    CylinderSceneBuilder().validate(specs, 3) shouldBe a[Left[?, ?]]

  "CylinderSceneBuilder.isCompatible" should "return true for two cylinders" in:
    val cylinder = ObjectSpec(objectType = "cylinder")
    CylinderSceneBuilder().isCompatible(cylinder, cylinder) shouldBe true

  it should "return false for a cylinder and a cone" in:
    CylinderSceneBuilder().isCompatible(
      ObjectSpec(objectType = "cylinder"), ObjectSpec(objectType = "cone")
    ) shouldBe false

  "CylinderSceneBuilder.calculateInstanceCount" should "be one instance per spec" in:
    val specs = List.fill(3)(ObjectSpec(objectType = "cylinder"))
    CylinderSceneBuilder().calculateInstanceCount(specs) shouldBe 3L

  "CylinderSceneBuilder.axisOf" should "centre a default cylinder on pos, size high, radius size/2" in:
    val spec = Cylinder(pos = Vec3(1f, 2f, 3f), size = 2f).toObjectSpec
    axisNear(
      CylinderSceneBuilder().axisOf(spec),
      Axis(start = (1f, 1f, 3f), end = (1f, 3f, 3f), radius = 1f)
    )

  it should "run the axis from p0 to p1, with the given radius" in:
    val spec = Cylinder(
      p0 = Some(Vec3(0f, -0.1f, 0f)), p1 = Some(Vec3(0f, 0f, 0f)), radius = Some(1.5f)
    ).toObjectSpec
    axisNear(
      CylinderSceneBuilder().axisOf(spec),
      Axis(start = (0f, -0.1f, 0f), end = (0f, 0f, 0f), radius = 1.5f)
    )

  it should "turn a default cylinder's axis by its rotation, about pos" in:
    // Rotating local +y by 90 deg about z sends it to -x.
    val spec = ObjectSpec(
      objectType = "cylinder", x = 1f, y = 2f, z = 3f,
      rotation = ObjectRotation(z = (math.Pi / 2).toFloat)
    )
    axisNear(
      CylinderSceneBuilder().axisOf(spec),
      Axis(start = (1.5f, 2f, 3f), end = (0.5f, 2f, 3f), radius = 0.5f)
    )

  it should "ignore pos, size and rotation when both end points are given" in:
    val spec = Cylinder(
      pos = Vec3(5f, 5f, 5f), size = 3f, rotation = Vec3(1f, 0f, 0f),
      p0 = Some(Vec3(0f, 0f, 0f)), p1 = Some(Vec3(0f, 1f, 0f)), radius = Some(0.2f)
    ).toObjectSpec
    axisNear(
      CylinderSceneBuilder().axisOf(spec),
      Axis(start = (0f, 0f, 0f), end = (0f, 1f, 0f), radius = 0.2f)
    )

  it should "place a cylinder from the command line where the same DSL cylinder goes" in:
    val cli = ObjectSpecParser.parse("type=cylinder:p0=0,-0.1,0:p1=0,0,0:radius=1.5")
      .fold(e => fail(e), identity)
    val dsl = Cylinder(
      p0 = Some(Vec3(0f, -0.1f, 0f)), p1 = Some(Vec3(0f, 0f, 0f)), radius = Some(1.5f)
    ).toObjectSpec
    CylinderSceneBuilder().axisOf(cli) shouldBe CylinderSceneBuilder().axisOf(dsl)
