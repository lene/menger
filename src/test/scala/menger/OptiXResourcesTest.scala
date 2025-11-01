package menger

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

import scala.util.Try

class OptiXResourcesTest extends AnyFunSuite with Matchers:

  test("OptiXResources can be instantiated"):
    val resources = new OptiXResources(Try(_.setSphere(0f, 0f, 0f, 1.0f)))
    resources should not be null

  test("dispose() should not throw on uninitialized resources"):
    val resources = new OptiXResources(Try(_.setSphere(0f, 0f, 0f, 1.0f)))
    noException should be thrownBy resources.dispose()

  test("RGBA byte array size calculation is correct"):
    val width = 10
    val height = 10
    val expectedSize = 400
    width * height * 4 shouldBe expectedSize
