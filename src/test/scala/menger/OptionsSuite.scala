package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers


class OptionsSuite extends AnyFlatSpec with Matchers:
  "empty options" should "give default timeout" in:
    val options = MengerCLIOptions(Seq[String]())
    options.timeout() shouldEqual 0

  "--timeout" should "set timeout" in:
    val options = MengerCLIOptions(Seq("--timeout", "1"))
    options.timeout() shouldEqual 1

  "--sponge-type" should "parse cube|square|tesseract|tesseract-sponge|tesseract-sponge-2" in:
    for spongeType <- Seq("cube", "square", "tesseract", "tesseract-sponge", "tesseract-sponge-2") do
      val options = MengerCLIOptions(Seq("--sponge-type", spongeType))
      options.spongeType() shouldEqual spongeType

  it should "default to square" in:
    val options = MengerCLIOptions(Seq[String]())
    options.spongeType() shouldEqual "square"

  it should "throw IllegalArgumentException if invalid" in:
    an [IllegalArgumentException] should be thrownBy MengerCLIOptions(Seq("--sponge-type", "invalid"))

  "--antialias-samples" should "set antialias samples" in:
    val options = MengerCLIOptions(Seq("--antialias-samples", "1"))
    options.antialiasSamples() shouldEqual 1

  "--projection-screen-w" should "be valid with matching --projection-eye-w" in:
    val options = MengerCLIOptions(Seq("--projection-screen-w", "1", "--projection-eye-w", "2"))
    options.projectionScreenW() shouldEqual 1
    options.projectionEyeW() shouldEqual 2

  it should "be invalid if <= 0" in:
    an [IllegalArgumentException] should be thrownBy
      MengerCLIOptions(Seq("--projection-screen-w", "0", "--projection-eye-w", "2"))

  it should "be invalid if <= --projection-screen-w)" in:
    an [IllegalArgumentException] should be thrownBy
      MengerCLIOptions(Seq("--projection-screen-w", "2", "--projection-eye-w", "2"))

  "--rot-x-w" should "be valid if 0 <= x < 360" in:
    val options = MengerCLIOptions(Seq("--rot-x-w", "1"))
    options.rotXW() shouldEqual 1

  it should "be invalid if >= 360" in:
    an [IllegalArgumentException] should be thrownBy MengerCLIOptions(Seq("--rot-x-w", "360"))

  "--animate" should "default to empty animation specifications" in:
    MengerCLIOptions(Seq()).animate() should equal (AnimationSpecifications())

  it should "be invalid for bad AnimationSpecifications syntax" in:
    an [IllegalArgumentException] should be thrownBy
      MengerCLIOptions(Seq("--animate", "0:1:2:3"))

  AnimationSpecification.TIMESCALE_PARAMETERS.foreach { timescale =>
    it should s"fail for a missing animated parameter when $timescale is specified" in:
      an[IllegalArgumentException] should be thrownBy
        MengerCLIOptions(Seq("--animate", s"$timescale=10"))
  }
  
  it should "fail if both frames and seconds are specified" in:
    an[IllegalArgumentException] should be thrownBy
      MengerCLIOptions(Seq("--animate", "frames=10:seconds=10"))
    
  it should "fail if no timescale is specified" in:
    an[IllegalArgumentException] should be thrownBy
      MengerCLIOptions(Seq("--animate", "rot-x=0-10"))

  AnimationSpecification.TIMESCALE_PARAMETERS.foreach { timescale =>
    AnimationSpecification.ALWAYS_VALID_PARAMETERS.foreach { parameter =>
      it should s"succeed when $timescale and $parameter are specified" in:
        MengerCLIOptions(Seq("--animate", s"$timescale=10:$parameter=0-10"))
    }
  }

  it should "fail if an invalid parameter is specified" in:
    an[IllegalArgumentException] should be thrownBy
      MengerCLIOptions(Seq("--animate", "frames=10:invalid=0-10"))

  Seq("square", "cube").foreach { sponge =>
    it should s"fail if a 4D parameter is specified for 3D sponge type $sponge" in :
      an[IllegalArgumentException] should be thrownBy
        MengerCLIOptions(Seq("--sponge-type", sponge, "--animate", "frames=10:rot-x-w=0-10"))
  }

  Seq("tesseract", "tesseract-sponge", "tesseract-sponge-2").foreach { sponge =>
    it should s"succeed if a 4D parameter is specified for 4D sponge type $sponge" in:
      MengerCLIOptions(Seq("--sponge-type", sponge, "--animate", "frames=10:rot-x-w=0-10"))
  }

  Seq("tesseract").foreach { sponge =>
    it should s"fail when level is specified for non-fractal sponge type $sponge" in:
      an[IllegalArgumentException] should be thrownBy
        MengerCLIOptions(Seq("--sponge-type", sponge, "--animate", s"frames=10:level=0-1"))
  }

  Seq("tesseract-sponge", "tesseract-sponge-2", "square", "cube").foreach { sponge =>
    it should s"succeed when level is specified for fractal sponge type $sponge" in:
      MengerCLIOptions(Seq("--sponge-type", sponge, "--animate", s"frames=10:level=0-1"))
  }

  it should "succeed when two valid animation specifications are given" in:
    MengerCLIOptions(Seq("--animate", "frames=10:rot-x=0-10", "--animate", "frames=10:rot-x=10-20"))

  it should "return the correct animation parameters"  in:
    val options = MengerCLIOptions(Seq("--animate", "frames=10:rot-x=0-10:rot-y=0-10"))
    options.animate().specification should have size 1
    options.animate().specification.head shouldEqual "frames=10:rot-x=0-10:rot-y=0-10"
    options.animate().parts should have size 1
    options.animate().parts.head.animationParameters should have size 2
    options.animate().parts.head.animationParameters("rot-x") shouldEqual (0, 10)
    options.animate().parts.head.animationParameters("rot-y") shouldEqual (0, 10)

  it should "return correct animation parameters when two valid animation specifications are given" in:
    val options = MengerCLIOptions(Seq("--animate", "frames=10:rot-x=0-10", "--animate", "frames=10:rot-x=10-20"))
    options.animate().specification should have size 2
    options.animate().parts should have size 2
    options.animate().parts.head.animationParameters should have size 1
    options.animate().parts.head.animationParameters("rot-x") shouldEqual (0, 10)
    options.animate().parts.last.animationParameters should have size 1
    options.animate().parts.last.animationParameters("rot-x") shouldEqual (10, 20)
