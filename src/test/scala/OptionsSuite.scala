import menger.MengerCLIOptions
import org.scalatest.Assertions.assertThrows
import org.scalatest.flatspec.AnyFlatSpec

class OptionsSuite extends AnyFlatSpec:
  "empty options" should "give default timeout" in:
    val options = MengerCLIOptions(Seq[String]())
    assert(options.timeout() == 0)

  "--timeout" should "set timeout" in:
    val options = MengerCLIOptions(Seq("--timeout", "1"))
    assert(options.timeout() == 1)

  "--sponge-type" should "parse cube|square|tesseract|tesseract-sponge|tesseract-sponge-2" in:
    for spongeType <- Seq("cube", "square", "tesseract", "tesseract-sponge", "tesseract-sponge-2") do
      val options = MengerCLIOptions(Seq("--sponge-type", spongeType))
      assert(options.spongeType() == spongeType)

  it should "default to square" in:
    val options = MengerCLIOptions(Seq[String]())
    assert(options.spongeType() == "square")

  it should "throw IllegalArgumentException if invalid" in:
    assertThrows[IllegalArgumentException]({
      MengerCLIOptions(Seq("--sponge-type", "invalid"))
    })

  "--antialias-samples" should "set antialias samples" in:
    val options = MengerCLIOptions(Seq("--antialias-samples", "1"))
    assert(options.antialiasSamples() == 1)


  "getConfig" should "return default config if no options" in:
    val options = MengerCLIOptions(Seq[String]())
    Main.getConfig(options)

  "--projection-screen-w" should "be valid with matching --projection-eye-w" in:
    val options = MengerCLIOptions(Seq("--projection-screen-w", "1", "--projection-eye-w", "2"))
    assert(options.projectionScreenW() == 1)
    assert(options.projectionEyeW() == 2)

  it should "be invalid if <= 0" in:
    assertThrows[IllegalArgumentException]({
      MengerCLIOptions(Seq("--projection-screen-w", "0", "--projection-eye-w", "2"))
    })

  it should "be invalid if <= --projection-screen-w)" in:
    assertThrows[IllegalArgumentException]({
      MengerCLIOptions(Seq("--projection-screen-w", "2", "--projection-eye-w", "2"))
    })

  "--rot-x-w" should "be valid if 0 <= x < 360" in:
    val options = MengerCLIOptions(Seq("--rot-x-w", "1"))
    assert(options.rotXW() == 1)

  it should "be invalid if >= 360" in:
    assertThrows[IllegalArgumentException]({
      MengerCLIOptions(Seq("--rot-x-w", "360"))
    })
