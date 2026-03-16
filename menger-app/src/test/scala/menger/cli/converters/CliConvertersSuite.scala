package menger.cli.converters

import com.badlogic.gdx.math.Vector3
import menger.cli.Axis
import menger.cli.PlaneSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CliConvertersSuite extends AnyFlatSpec with Matchers:

  private def singleArg(value: String): List[(String, List[String])] =
    List(("", List(value)))

  "vector3Converter" should "parse valid x,y,z string" in:
    val result = vector3Converter.parse(singleArg("0,0,3"))
    result shouldBe Right(Some(Vector3(0f, 0f, 3f)))

  it should "parse negative and fractional values" in:
    val result = vector3Converter.parse(singleArg("1.5,-2.0,0"))
    result shouldBe Right(Some(Vector3(1.5f, -2.0f, 0f)))

  it should "return Right(None) for empty input" in:
    vector3Converter.parse(List.empty) shouldBe Right(None)

  it should "return Left for wrong number of components" in:
    vector3Converter.parse(singleArg("1,2")).isLeft shouldBe true

  it should "return Left for non-numeric values" in:
    vector3Converter.parse(singleArg("x,y,z")).isLeft shouldBe true

  "planeSpecConverter" should "parse y:-2" in:
    planeSpecConverter.parse(singleArg("y:-2")) shouldBe
      Right(Some(PlaneSpec(Axis.Y, positive = true, -2f)))

  it should "parse +z:5.5" in:
    planeSpecConverter.parse(singleArg("+z:5.5")) shouldBe
      Right(Some(PlaneSpec(Axis.Z, positive = true, 5.5f)))

  it should "parse -x:3 with negative direction" in:
    planeSpecConverter.parse(singleArg("-x:3")) shouldBe
      Right(Some(PlaneSpec(Axis.X, positive = false, 3f)))

  it should "return Right(None) for empty input" in:
    planeSpecConverter.parse(List.empty) shouldBe Right(None)

  it should "return Left for invalid format" in:
    planeSpecConverter.parse(singleArg("foo")).isLeft shouldBe true

  it should "return Left for invalid axis" in:
    planeSpecConverter.parse(singleArg("w:3")).isLeft shouldBe true

  "planeColorSpecConverter" should "parse solid hex color" in:
    val result = planeColorSpecConverter.parse(singleArg("FF0000"))
    result.isRight shouldBe true
    result.toOption.flatten.map(_.isSolid) shouldBe Some(true)

  it should "parse hex color with # prefix" in:
    val result = planeColorSpecConverter.parse(singleArg("#FF0000"))
    result.isRight shouldBe true
    result.toOption.flatten.map(_.isSolid) shouldBe Some(true)

  it should "parse checkered hex colors" in:
    val result = planeColorSpecConverter.parse(singleArg("FF0000:0000FF"))
    result.isRight shouldBe true
    result.toOption.flatten.map(_.isCheckered) shouldBe Some(true)

  it should "return Right(None) for empty input" in:
    planeColorSpecConverter.parse(List.empty) shouldBe Right(None)

  it should "return Left for invalid hex digits" in:
    planeColorSpecConverter.parse(singleArg("ZZZ000")).isLeft shouldBe true

  it should "return Left for wrong hex length" in:
    planeColorSpecConverter.parse(singleArg("FF00")).isLeft shouldBe true
