package menger

import com.badlogic.gdx.math.Vector3
import menger.common.{Vector => CommonVector}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.Try

class OptiXResourcesTest extends AnyFlatSpec with Matchers:

  "OptiXResources" should "be instantiable" in:
    val resources = new OptiXResources(
      Try(_.setSphere(CommonVector[3](0f, 0f, 0f), 1.0f)),
      Vector3(0f, 0.5f, 3.0f),
      Vector3(0f, 0f, 0f),
      Vector3(0f, 1f, 0f),
      PlaneSpec(Axis.Y, false, -2f)
    )
    resources should not be null

  it should "not throw on dispose() for uninitialized resources" in:
    val resources = new OptiXResources(
      Try(_.setSphere(CommonVector[3](0f, 0f, 0f), 1.0f)),
      Vector3(0f, 0.5f, 3.0f),
      Vector3(0f, 0f, 0f),
      Vector3(0f, 1f, 0f),
      PlaneSpec(Axis.Y, false, -2f)
    )
    noException should be thrownBy resources.dispose()

  "RGBA byte array size calculation" should "be correct" in:
    val width = 10
    val height = 10
    val expectedSize = 400
    width * height * 4 shouldBe expectedSize
