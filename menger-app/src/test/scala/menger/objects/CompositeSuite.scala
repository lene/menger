package menger.objects

import scala.util.Failure
import scala.util.Try

import menger.RotationProjectionParameters
import menger.common.UnknownGeometryException
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CompositeSuite extends AnyFlatSpec with Matchers:
  given menger.common.ProfilingConfig = menger.common.ProfilingConfig.disabled

  "Composite toString" should "show component geometries" in:
    val sphere = Sphere()
    val cube = Cube()
    val composite = Composite(geometries = List(sphere, cube))
    composite.toString should be("Composite(Sphere, Cube)")

  "Composite.parseCompositeFromCLIOption" should "succeed for composite[cube,sphere] type" in:
    val alwaysSucceed = (_: String, _: Float, _: Any, _: Int, _: RotationProjectionParameters) =>
      Try(Sphere(): Geometry)
    val result = Composite.parseCompositeFromCLIOption(
      "composite[cube,sphere]", 1f, null, 0, RotationProjectionParameters(), alwaysSucceed
    )
    result.isSuccess shouldBe true
    result.get shouldBe a[Composite]

  it should "fail for non-composite type" in:
    val alwaysSucceed = (_: String, _: Float, _: Any, _: Int, _: RotationProjectionParameters) =>
      Try(Sphere(): Geometry)
    val result = Composite.parseCompositeFromCLIOption(
      "cube", 1f, null, 0, RotationProjectionParameters(), alwaysSucceed
    )
    result.isFailure shouldBe true
    result.failed.get shouldBe a[UnknownGeometryException]

  it should "propagate geometry creation failure" in:
    val alwaysFail = (_: String, _: Float, _: Any, _: Int, _: RotationProjectionParameters) =>
      Failure[Geometry](RuntimeException("geometry failed"))
    val result = Composite.parseCompositeFromCLIOption(
      "composite[cube,sphere]", 1f, null, 0, RotationProjectionParameters(), alwaysFail
    )
    result.isFailure shouldBe true

