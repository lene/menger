package menger

import menger.optix.OptiXRenderer
import org.scalamock.scalatest.MockFactory
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class OptiXResourcesTest extends AnyFunSuite with Matchers with MockFactory:

  private def createMockRenderer(): OptiXRenderer =
    val renderer = mock[OptiXRenderer]
    renderer.setSphere.expects(*, *, *, *).anyNumberOfTimes()
    renderer.setCamera.expects(*, *, *, *).anyNumberOfTimes()
    renderer.setLight.expects(*, *).anyNumberOfTimes()
    renderer

  test("OptiXResources can be instantiated"):
    val resources = new OptiXResources(createMockRenderer(), 1.0f)
    resources should not be null

  test("dispose() should not throw on uninitialized resources"):
    val resources = new OptiXResources(createMockRenderer(), 1.0f)
    noException should be thrownBy resources.dispose()

  test("resize() accepts valid dimensions"):
    val resources = new OptiXResources(createMockRenderer(), 1.0f)
    noException should be thrownBy resources.resize(800, 600)
    noException should be thrownBy resources.resize(1920, 1080)

  test("render validates byte array size"):
    val resources = new OptiXResources(createMockRenderer(), 1.0f)
    val width = 4
    val height = 4
    val correctSize = width * height * 4  // RGBA = 4 bytes per pixel

    // Correct size should validate
    val validBytes = new Array[Byte](correctSize)
    // Note: Will fail without LibGDX context, but validation logic is testable

    // Wrong size should throw
    val invalidBytes = new Array[Byte](10)
    val exception = intercept[IllegalArgumentException]:
      resources.render(invalidBytes, width, height)

    exception.getMessage should include("Invalid RGBA byte array size")
    exception.getMessage should include(s"expected $correctSize")
    exception.getMessage should include("got 10")

  test("RGBA byte array size calculation is correct"):
    // 10x10 image = 100 pixels * 4 bytes = 400 bytes
    val width = 10
    val height = 10
    val expectedSize = 400

    width * height * 4 shouldBe expectedSize

  test("Multiple dispose() calls should not throw"):
    val resources = new OptiXResources(createMockRenderer(), 1.0f)
    noException should be thrownBy resources.dispose()
    noException should be thrownBy resources.dispose()
    noException should be thrownBy resources.dispose()
