package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.rogach.scallop.exceptions.ScallopException
import com.badlogic.gdx.graphics.Color


class OptionsSuite extends AnyFlatSpec with Matchers:
  
  class SafeMengerCLIOptions(args: Seq[String]) extends menger.MengerCLIOptions(args):
    @SuppressWarnings(Array("org.wartremover.warts.Throw"))
    override def onError(e: Throwable): Unit = throw e

  "empty options" should "give default timeout" in:
    val options = SafeMengerCLIOptions(Seq[String]())
    options.timeout() shouldEqual 0

  "--timeout" should "set timeout" in:
    val options = SafeMengerCLIOptions(Seq("--timeout", "1"))
    options.timeout() shouldEqual 1

  "--sponge-type" should "accept basic 3D shapes" in:
    for spongeType <- Seq("cube", "square") do
      val options = SafeMengerCLIOptions(Seq("--sponge-type", spongeType))
      options.spongeType() shouldEqual spongeType

  it should "accept 4D shapes when standalone" in:
    for spongeType <- Seq("tesseract", "tesseract-sponge", "tesseract-sponge-2") do
      val options = SafeMengerCLIOptions(Seq("--sponge-type", spongeType))
      options.spongeType() shouldEqual spongeType

  it should "accept new sponge types" in:
    for spongeType <- Seq("square-sponge", "cube-sponge") do
      val options = SafeMengerCLIOptions(Seq("--sponge-type", spongeType))
      options.spongeType() shouldEqual spongeType

  it should "accept simple composites with 3D shapes" in:
    for composite <- Seq("composite[cube,square]", "composite[cube]", "composite[square]") do
      val options = SafeMengerCLIOptions(Seq("--sponge-type", composite))
      options.spongeType() shouldEqual composite

  it should "reject nested composites with only 3D shapes" in:
    for composite <- Seq("composite[composite[cube,square],cube]", "composite[cube,composite[square]]") do
      an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--sponge-type", composite))

  it should "reject composites with 4D shapes" in:
    for composite <- Seq("composite[cube,tesseract]", "composite[tesseract]", "composite[tesseract-sponge]") do
      an [ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--sponge-type", composite))

  it should "reject nested composites containing 4D shapes" in:
    for composite <- Seq("composite[composite[cube,tesseract],cube]", "composite[cube,composite[tesseract]]") do
      an [ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--sponge-type", composite))

  it should "default to square" in:
    val options = SafeMengerCLIOptions(Seq[String]())
    options.spongeType() shouldEqual "square"

  it should "reject invalid sponge types" in:
    for invalid <- Seq("invalid", "composite[invalid]", "composite[]") do
      an [ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--sponge-type", invalid))

  it should "reject malformed composite syntax" in:
    for malformed <- Seq("composite[cube", "compositecube,square]", "composite") do
      an [ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--sponge-type", malformed))

  "--antialias-samples" should "set antialias samples" in:
    val options = SafeMengerCLIOptions(Seq("--antialias-samples", "1"))
    options.antialiasSamples() shouldEqual 1

  "--projection-screen-w" should "be valid with matching --projection-eye-w" in:
    val options = SafeMengerCLIOptions(Seq("--projection-screen-w", "1", "--projection-eye-w", "2"))
    options.projectionScreenW() shouldEqual 1
    options.projectionEyeW() shouldEqual 2

  it should "be invalid if <= 0" in:
    an [ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--projection-screen-w", "0", "--projection-eye-w", "2"))

  it should "be invalid if <= --projection-screen-w)" in:
    an [ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--projection-screen-w", "2", "--projection-eye-w", "2"))

  "rotation options" should "be valid if 0 <= x < 360" in:
    for opt <- Seq("--rot-x", "--rot-y", "--rot-z", "--rot-x-w", "--rot-y-w", "--rot-z-w") do
      val options = SafeMengerCLIOptions(Seq(opt, "1"))
      opt match
        case "--rot-x" => options.rotX() shouldEqual 1
        case "--rot-y" => options.rotY() shouldEqual 1
        case "--rot-z" => options.rotZ() shouldEqual 1
        case "--rot-x-w" => options.rotXW() shouldEqual 1
        case "--rot-y-w" => options.rotYW() shouldEqual 1
        case "--rot-z-w" => options.rotZW() shouldEqual 1

  it should "be invalid if >= 360" in:
    for opt <- Seq("--rot-x", "--rot-y", "--rot-z", "--rot-x-w", "--rot-y-w", "--rot-z-w") do
      an [ScallopException] should be thrownBy SafeMengerCLIOptions(Seq(opt, "360"))

  "--animate" should "default to empty animation specifications" in:
    SafeMengerCLIOptions(Seq()).animate.toOption shouldBe None

  it should "be invalid for bad AnimationSpecifications syntax" in:
    an [ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--animate", "0:1:2:3"))

  AnimationSpecification.TIMESCALE_PARAMETERS.foreach { timescale =>
    it should s"fail for a missing animated parameter when $timescale is specified" in:
      an[ScallopException] should be thrownBy
        SafeMengerCLIOptions(Seq("--animate", s"$timescale=10"))
  }
  
  it should "fail if both frames and seconds are specified" in:
    an[IllegalArgumentException] should be thrownBy
      SafeMengerCLIOptions(Seq("--animate", "frames=10:seconds=10"))
    
  it should "fail if no timescale is specified" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--animate", "rot-x=0-10"))

  AnimationSpecification.TIMESCALE_PARAMETERS.foreach { timescale =>
    AnimationSpecification.ALWAYS_VALID_PARAMETERS.foreach { parameter =>
      it should s"succeed when $timescale and $parameter are specified" in:
        SafeMengerCLIOptions(Seq("--animate", s"$timescale=10:$parameter=0-10"))
    }
  }

  it should "fail if an invalid parameter is specified" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--animate", "frames=10:invalid=0-10"))

  Seq("square", "cube").foreach { sponge =>
    it should s"fail if a 4D parameter is specified for 3D sponge type $sponge" in :
      an[ScallopException] should be thrownBy
        SafeMengerCLIOptions(Seq("--sponge-type", sponge, "--animate", "frames=10:rot-x-w=0-10"))
  }

  Seq("tesseract", "tesseract-sponge", "tesseract-sponge-2").foreach { sponge =>
    it should s"succeed if a 4D parameter is specified for 4D sponge type $sponge" in:
      SafeMengerCLIOptions(Seq("--sponge-type", sponge, "--animate", "frames=10:rot-x-w=0-10"))
  }

  Seq("tesseract").foreach { sponge =>
    it should s"fail when level is specified for non-fractal sponge type $sponge" in:
      an[ScallopException] should be thrownBy
        SafeMengerCLIOptions(Seq("--sponge-type", sponge, "--animate", s"frames=10:level=0-1"))
  }

  Seq("tesseract-sponge", "tesseract-sponge-2", "square", "cube").foreach { sponge =>
    it should s"succeed when level is specified for fractal sponge type $sponge" in:
      SafeMengerCLIOptions(Seq("--sponge-type", sponge, "--animate", s"frames=10:level=0-1"))
  }

  it should "succeed when two valid animation specifications are given" in:
    SafeMengerCLIOptions(Seq("--animate", "frames=10:rot-x=0-10", "--animate", "frames=10:rot-x=10-20"))

  it should "return the correct animation parameters"  in:
    val options = SafeMengerCLIOptions(Seq("--animate", "frames=10:rot-x=0-10:rot-y=0-10"))
    options.animate().specification should have size 1
    options.animate().specification.head shouldEqual "frames=10:rot-x=0-10:rot-y=0-10"
    options.animate().parts should have size 1
    options.animate().parts.head.animationParameters should have size 2
    options.animate().parts.head.animationParameters("rot-x") shouldEqual (0, 10)
    options.animate().parts.head.animationParameters("rot-y") shouldEqual (0, 10)

  it should "return correct animation parameters when two valid animation specifications are given" in:
    val options = SafeMengerCLIOptions(Seq("--animate", "frames=10:rot-x=0-10", "--animate", "frames=10:rot-x=10-20"))
    options.animate().specification should have size 2
    options.animate().parts should have size 2
    options.animate().parts.head.animationParameters should have size 1
    options.animate().parts.head.animationParameters("rot-x") shouldEqual (0, 10)
    options.animate().parts.last.animationParameters should have size 1
    options.animate().parts.last.animationParameters("rot-x") shouldEqual (10, 20)

  it should "fail if both frames and seconds are specified in separate specifications" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--animate", "frames=10:rot-x=0-10", "--animate", "seconds=10:rot-x=10-20"))

  it should "fail if the same rotation is declared as static and animated" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--animate", "frames=10:rot-x=0-10", "--rot-x", "10"))

  it should "fail if level is declared both as static and animated" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--animate", "frames=10:level=0-2", "--level", "2"))

  it should "fail if level is explicitly set to default value and animated" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--animate", "frames=10:level=0-2", "--level", "1.0"))

  it should "succeed if level is animated but not explicitly set as option" in:
    SafeMengerCLIOptions(Seq("--animate", "frames=10:level=0-2", "--sponge-type", "cube"))

  it should "succeed if level is set as option but not animated" in:
    SafeMengerCLIOptions(Seq("--level", "2", "--sponge-type", "cube"))

  "color option" should "default color to light gray" in:
    val options = SafeMengerCLIOptions(Seq[String]())
    options.color() shouldEqual Color.LIGHT_GRAY

  "color option" should "accept RGB hex codes (6 digits)" in :
    SafeMengerCLIOptions(Seq("--color", "ff0000")).color() shouldEqual new Color(1f, 0f, 0f, 1f)
    SafeMengerCLIOptions(Seq("--color", "00ff00")).color() shouldEqual new Color(0f, 1f, 0f, 1f)
    SafeMengerCLIOptions(Seq("--color", "0000ff")).color() shouldEqual new Color(0f, 0f, 1f, 1f)

  it should "accept RGBA hex codes (8 digits)" in :
    SafeMengerCLIOptions(Seq("--color", "ff00007f")).color() shouldEqual new Color(1f, 0f, 0f, 0.5f)
    SafeMengerCLIOptions(Seq("--color", "00ff007f")).color() shouldEqual new Color(0f, 1f, 0f, 0.5f)
    SafeMengerCLIOptions(Seq("--color", "0000ff7f")).color() shouldEqual new Color(0f, 0f, 1f, 0.5f)

  it should "fail for invalid hex codes" in :
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--color", "gg0000"))

  it should "fail for too short hex codes" in:
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--color", "0000"))

  it should "fail for too long hex codes" in:
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--color", "000000000"))

  it should "accept RGB integer triplets" in :
    SafeMengerCLIOptions(Seq("--color", "255,0,0")).color() shouldEqual new Color(1f, 0f, 0f, 1f)
    SafeMengerCLIOptions(Seq("--color", "0,255,0")).color() shouldEqual new Color(0f, 1f, 0f, 1f)
    SafeMengerCLIOptions(Seq("--color", "0,0,255")).color() shouldEqual new Color(0f, 0f, 1f, 1f)

  it should "accept RGBA integer quadruplets" in :
    SafeMengerCLIOptions(Seq("--color", "255,0,0,128")).color() shouldEqual new Color(1f, 0f, 0f, 128f / 255f)
    SafeMengerCLIOptions(Seq("--color", "0,255,0,128")).color() shouldEqual new Color(0f, 1f, 0f, 128f / 255f)
    SafeMengerCLIOptions(Seq("--color", "0,0,255,128")).color() shouldEqual new Color(0f, 0f, 1f, 128f / 255f)

  it should "accept an alpha value of zero" in:
    SafeMengerCLIOptions(Seq("--color", "ff000000")).color() shouldEqual new Color(1f, 0f, 0f, 0f)
    SafeMengerCLIOptions(Seq("--color", "00ff0000")).color() shouldEqual new Color(0f, 1f, 0f, 0f)
    SafeMengerCLIOptions(Seq("--color", "0000ff00")).color() shouldEqual new Color(0f, 0f, 1f, 0f)
    SafeMengerCLIOptions(Seq("--color", "255,0,0,0")).color() shouldEqual new Color(1f, 0f, 0f, 0f)
    SafeMengerCLIOptions(Seq("--color", "0,255,0,0")).color() shouldEqual new Color(0f, 1f, 0f, 0f)
    SafeMengerCLIOptions(Seq("--color", "0,0,255,0")).color() shouldEqual new Color(0f, 0f, 1f, 0f)

  it should "fail for integer values out of range" in :
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--color", "256,0,0"))
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--color", "0,-1,255"))
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--color", "0,0,255,300"))

  it should "fail for too short integer triplets (aka pairs)" in:
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--color", "0,0"))

  it should "fail for too long integer quadruplets (aka quintuplets)" in:
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--color", "0,0,255,255,0"))

  it should "fail for integer triplets with trailing commas" in:
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--color", "0,0,255,"))

  it should "fail for integer triplets with missing parts" in:
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--color", "0,,255"))

  it should "fail if any member is not an integer" in:
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--color", "0,0,blue"))

  it should "fail for unrecognized color names" in :
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--color", "red"))
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--color", "whatever"))

  "face-color and line-color options" should "work together for overlay mode" in:
    val options = SafeMengerCLIOptions(Seq("--face-color", "ffffff80", "--line-color", "000000ff"))
    options.faceColor.toOption shouldBe defined
    options.lineColor.toOption shouldBe defined
    options.faceColor() shouldEqual Color.valueOf("ffffff80")
    options.lineColor() shouldEqual new Color(0f, 0f, 0f, 1f)

  it should "fail if --color is used with --face-color" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--color", "ff0000", "--face-color", "ffffff80", "--line-color", "000000ff"))

  it should "fail if --color is used with --line-color" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--color", "ff0000", "--line-color", "000000ff"))

  it should "fail if only --face-color is specified" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--face-color", "ffffff80"))

  it should "fail if only --line-color is specified" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--line-color", "000000ff"))

  it should "fail if --lines is used with --face-color" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--lines", "--face-color", "ffffff80", "--line-color", "000000ff"))

  it should "fail if --lines is used with --line-color" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--lines", "--line-color", "000000ff"))

  it should "succeed with --color and --lines together" in:
    SafeMengerCLIOptions(Seq("--color", "ff0000", "--lines"))

  "--sponge-type sphere" should "be accepted" in:
    val options = SafeMengerCLIOptions(Seq("--sponge-type", "sphere", "--optix"))
    options.spongeType() shouldEqual "sphere"

  "--optix" should "parse correctly and default to false" in:
    val options1 = SafeMengerCLIOptions(Seq())
    options1.optix() shouldEqual false

    val options2 = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere"))
    options2.optix() shouldEqual true

  "--radius" should "accept positive floats" in:
    val options1 = SafeMengerCLIOptions(Seq("--radius", "1.5"))
    options1.radius() shouldEqual 1.5f

    val options2 = SafeMengerCLIOptions(Seq("--radius", "0.5"))
    options2.radius() shouldEqual 0.5f

    val options3 = SafeMengerCLIOptions(Seq("--radius", "10.0"))
    options3.radius() shouldEqual 10.0f

  it should "default to 1.0" in:
    val options = SafeMengerCLIOptions(Seq())
    options.radius() shouldEqual 1.0f

  it should "reject zero" in:
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--radius", "0"))

  it should "reject negative values" in:
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(Seq("--radius", "-1.0"))

  "OptiX validation" should "fail when sponge-type sphere without --optix" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--sponge-type", "sphere"))

  it should "fail when --optix without sponge-type sphere" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix"))

    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "cube"))

  it should "succeed when both --optix and sponge-type sphere" in:
    val options = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere"))
    options.optix() shouldEqual true
    options.spongeType() shouldEqual "sphere"

  "MengerCLIOptions --stats" should "parse when provided" in :
    val opts = SafeMengerCLIOptions(List("--optix", "--sponge-type", "sphere", "--stats"))
    opts.stats() shouldBe true
  
  it should "be optional" in :
    val opts = SafeMengerCLIOptions(List())
    opts.stats.toOption shouldBe Some(false)
  
  it should "have default value false" in :
    val opts = SafeMengerCLIOptions(List())
    opts.stats() shouldBe false
  
  it should "be false without explicit flag" in :
    val opts = SafeMengerCLIOptions(List("--optix", "--sponge-type", "sphere"))
    opts.stats() shouldBe false
  
  "Main.createEngine" should "create OptiXEngine with stats disabled when option not provided" in :
    val opts = SafeMengerCLIOptions(List("--optix", "--sponge-type", "sphere"))
    val enableStats = opts.stats()
  
    enableStats shouldBe false
  
  it should "create OptiXEngine with stats enabled when option provided" in :
    val opts = SafeMengerCLIOptions(List("--optix", "--sponge-type", "sphere", "--stats"))
    val enableStats = opts.stats()
  
    enableStats shouldBe true
