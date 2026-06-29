package menger.engines.scene

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.CurveData
import menger.ObjectRotation
import menger.ObjectSpec

class CurveSceneBuilderSuite extends AnyFlatSpec with Matchers:

  private def curveSpec(
    points: Seq[Float] = Seq(0f, 0f, 0f, 1f, 0f, 0f, 1f, 1f, 0f, 0f, 1f, 0f),
    widths: Seq[Float] = Seq(0.05f, 0.05f, 0.05f, 0.05f)
  ): ObjectSpec =
    ObjectSpec(
      objectType = "curve",
      curveData = Some(
        menger.CurveData(points = points.toVector, widths = widths.toVector)
      )
    )

  "CurveSceneBuilder.validate" should "accept valid curve specs" in:
    val builder = CurveSceneBuilder()
    val result = builder.validate(List(curveSpec()), maxInstances = 8)
    result shouldBe Right(())

  it should "reject empty spec list" in:
    val builder = CurveSceneBuilder()
    builder.validate(List.empty, maxInstances = 8).isLeft shouldBe true

  it should "reject unsupported fields like texture" in:
    val builder = CurveSceneBuilder()
    val spec = curveSpec().copy(texture = Some("wood.png"))
    builder.validate(List(spec), maxInstances = 8) match
      case Left(msg) => msg should include("texture")
      case Right(_)  => fail("Expected validation failure for texture on curve")

  it should "reject rotation on curves" in:
    val builder = CurveSceneBuilder()
    val spec = curveSpec().copy(rotation = ObjectRotation(45f))
    builder.validate(List(spec), maxInstances = 8) match
      case Left(msg) => msg should include("rotation")
      case Right(_)  => fail("Expected validation failure for rotation on curve")

  it should "reject proceduralType on curves" in:
    val builder = CurveSceneBuilder()
    val spec = curveSpec().copy(procedural = menger.ProceduralSpec(1, 1.0f))
    builder.validate(List(spec), maxInstances = 8) match
      case Left(msg) => msg should include("procedural")
      case Right(_)  => fail("Expected validation failure for procedural on curve")

  it should "accept curves with color and material" in:
    val builder = CurveSceneBuilder()
    val spec = curveSpec().copy(
      color = Some(menger.common.Color(1f, 0.5f, 0f)),
      ior = 1.5f
    )
    val result = builder.validate(List(spec), maxInstances = 8)
    result shouldBe Right(())
