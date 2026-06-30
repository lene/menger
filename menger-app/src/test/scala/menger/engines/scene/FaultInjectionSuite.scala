package menger.engines.scene

import menger.CurveData
import menger.ObjectSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class FaultInjectionSuite extends AnyFlatSpec with Matchers:

  private def spec(s: String): ObjectSpec = ObjectSpec.parse(s).toOption.get

  // ── InstanceId.fromNative ──────────────────────────────────────────────────

  "InstanceId.fromNative" should "throw RuntimeException on -1" in:
    val ex = intercept[RuntimeException] {
      InstanceId.fromNative(-1, "test operation")
    }
    ex.getMessage should include("Native renderer failed to add test operation")

  it should "return the ID for zero" in:
    val id = InstanceId.fromNative(0, "zero allocation")
    InstanceId.raw(id) shouldBe 0

  it should "return the ID for a positive value" in:
    val id = InstanceId.fromNative(42, "positive allocation")
    InstanceId.raw(id) shouldBe 42

  "InstanceId opaque type" should "round-trip through raw" in:
    val id = InstanceId.fromNative(7, "round trip")
    InstanceId.raw(id) shouldBe 7

  // ── Validation fail-fast: common cases ─────────────────────────────────────

  "SceneBuilder validate" should "reject empty specs" in:
    val builders = List(
      SphereSceneBuilder(),
      CurveSceneBuilder(),
      ConeSceneBuilder(),
      PlaneSceneBuilder(),
      CubeSpongeSceneBuilder(),
      LSystemSceneBuilder(),
      Menger4DSceneBuilder(),
      Sierpinski4DSceneBuilder(),
      Hexadecachoron4DSceneBuilder()
    )
    builders.foreach { builder =>
      val result = builder.validate(List.empty, maxInstances = 64)
      result shouldBe a[Left[String, Unit]]
      result.left.getOrElse("") should include("empty")
    }

  it should "reject too many instances" in:
    val builder = SphereSceneBuilder()
    val specs = List.fill(10)(ObjectSpec("sphere"))
    val result = builder.validate(specs, maxInstances = 5)
    result shouldBe a[Left[String, Unit]]
    result.left.getOrElse("") should include("exceeds max instances")

  // ── Per-builder type mismatch ──────────────────────────────────────────────

  "SphereSceneBuilder" should "reject non-sphere types" in:
    val builder = SphereSceneBuilder()
    val result = builder.validate(List(ObjectSpec("cube")), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]
    result.left.getOrElse("") should include("spheres")

  "CurveSceneBuilder" should "reject specs without curveData" in:
    val builder = CurveSceneBuilder()
    val noData = ObjectSpec(objectType = "curve")
    val result = builder.validate(List(noData), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]
    result.left.getOrElse("") should include("curveData")

  it should "reject non-curve types" in:
    val builder = CurveSceneBuilder()
    val result = builder.validate(List(ObjectSpec("sphere")), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]

  "ConeSceneBuilder" should "reject non-cone types" in:
    val builder = ConeSceneBuilder()
    val result = builder.validate(List(ObjectSpec("sphere")), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]

  "PlaneSceneBuilder" should "reject non-plane types" in:
    val builder = PlaneSceneBuilder()
    val result = builder.validate(List(ObjectSpec("sphere")), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]

  "CubeSpongeSceneBuilder" should "reject specs without level" in:
    val builder = CubeSpongeSceneBuilder()
    val noLevel = ObjectSpec(objectType = "cube-sponge")
    val result = builder.validate(List(noLevel), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]
    result.left.getOrElse("") should include("level")

  it should "reject non-cube-sponge types" in:
    val builder = CubeSpongeSceneBuilder()
    val result = builder.validate(List(ObjectSpec("sphere")), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]

  "Menger4DSceneBuilder" should "reject specs without level" in:
    val builder = Menger4DSceneBuilder()
    val noLevel = ObjectSpec(objectType = "menger4d")
    val result = builder.validate(List(noLevel), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]

  it should "reject non-menger4d types" in:
    val builder = Menger4DSceneBuilder()
    val result = builder.validate(List(ObjectSpec("sphere")), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]

  "Sierpinski4DSceneBuilder" should "reject specs without level" in:
    val builder = Sierpinski4DSceneBuilder()
    val noLevel = ObjectSpec(objectType = "sierpinski4d")
    val result = builder.validate(List(noLevel), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]

  it should "reject non-sierpinski4d types" in:
    val builder = Sierpinski4DSceneBuilder()
    val result = builder.validate(List(ObjectSpec("sphere")), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]

  "Hexadecachoron4DSceneBuilder" should "reject specs without level" in:
    val builder = Hexadecachoron4DSceneBuilder()
    val noLevel = ObjectSpec(objectType = "hexadecachoron4d")
    val result = builder.validate(List(noLevel), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]

  it should "reject non-hexadecachoron4d types" in:
    val builder = Hexadecachoron4DSceneBuilder()
    val result = builder.validate(List(ObjectSpec("sphere")), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]

  "LSystemSceneBuilder" should "reject non-lsystem types" in:
    val builder = LSystemSceneBuilder()
    val result = builder.validate(List(ObjectSpec("sphere")), maxInstances = 64)
    result shouldBe a[Left[String, Unit]]

  // ── Valid specs pass validation ────────────────────────────────────────────

  "Valid specs" should "pass validation for each builder" in:
    val curveSpec = ObjectSpec(
      objectType = "curve",
      curveData = Some(CurveData(
        points = Vector(0f, 0f, 0f, 1f, 0f, 0f),
        widths = Vector(0.05f, 0.05f)
      ))
    )
    val builders = List(
      (SphereSceneBuilder(), ObjectSpec("sphere")),
      (CurveSceneBuilder(), curveSpec),
      (ConeSceneBuilder(), ObjectSpec("cone")),
      (PlaneSceneBuilder(), ObjectSpec("plane")),
      (LSystemSceneBuilder(), ObjectSpec("lsystem"))
    )
    builders.foreach { case (builder, validSpec) =>
      val result = builder.validate(List(validSpec), maxInstances = 64)
      result shouldBe Right(())
    }
