package menger.objects

import scala.util.Failure

import menger.common.UnknownGeometryException
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GeometryFactorySuite extends AnyFlatSpec with Matchers:
  given menger.ProfilingConfig = menger.ProfilingConfig.disabled

  "GeometryFactory.supportedTypes" should "contain all documented geometry types" in:
    GeometryFactory.supportedTypes should contain("square")
    GeometryFactory.supportedTypes should contain("cube")
    GeometryFactory.supportedTypes should contain("square-sponge")
    GeometryFactory.supportedTypes should contain("cube-sponge")
    GeometryFactory.supportedTypes should contain("tesseract")
    GeometryFactory.supportedTypes should contain("tesseract-sponge")
    GeometryFactory.supportedTypes should contain("tesseract-sponge-2")
    GeometryFactory.supportedTypes should contain("sphere")

  "GeometryFactory.isValidType" should "return true for all supported types" in:
    for t <- GeometryFactory.supportedTypes do
      GeometryFactory.isValidType(t) shouldBe true

  it should "return true for composite[] prefix" in:
    GeometryFactory.isValidType("composite[cube,sphere]") shouldBe true

  it should "return false for unknown type" in:
    GeometryFactory.isValidType("unknown-geometry") shouldBe false

  it should "return false for empty string" in:
    GeometryFactory.isValidType("") shouldBe false

  "GeometryFactory.create" should "fail with UnknownGeometryException for unknown type" in:
    val result = GeometryFactory.create("not-a-type", 1f, null, 0, menger.RotationProjectionParameters())
    result shouldBe a[Failure[?]]
    result.failed.get shouldBe a[UnknownGeometryException]

  it should "fail for sphere type (use --objects type=sphere for OptiX sphere rendering)" in:
    val result = GeometryFactory.create("sphere", 1f, null, 0, menger.RotationProjectionParameters())
    result shouldBe a[Failure[?]]
    result.failed.get shouldBe a[UnsupportedOperationException]
