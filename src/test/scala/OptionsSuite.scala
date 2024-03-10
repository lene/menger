import menger.MengerCLIOptions
import org.rogach.scallop.exceptions.ScallopException
import org.scalatest.funsuite.AnyFunSuite

class OptionsSuite extends AnyFunSuite:
  test("empty options give default timeout") {
    val options = MengerCLIOptions(Seq[String]())
    assert(options.timeout() == 0)
  }

  test("--timeout option sets timeout") {
    val options = MengerCLIOptions(Seq("--timeout", "1"))
    assert(options.timeout() == 1)
  }

  test("--sponge-type cube|square|tesseract") {
    Seq("cube", "square", "tesseract").foreach { spongeType =>
      val options = MengerCLIOptions(Seq("--sponge-type", spongeType))
      assert(options.spongeType() == spongeType)
    }
  }

  test("default for --sponge-type") {
    val options = MengerCLIOptions(Seq[String]())
    assert(options.spongeType() == "square")
  }


  test("invalid --sponge-type throws IllegalArgumentException") {
    assertThrows[IllegalArgumentException]({
      MengerCLIOptions(Seq("--sponge-type", "invalid"))
    })
  }

  test("--antialias-samples") {
    val options = MengerCLIOptions(Seq("--antialias-samples", "1"))
    assert(options.antialiasSamples() == 1)
  }

  test("getConfig") {
    val options = MengerCLIOptions(Seq[String]())
    Main.getConfig(options)
  }

  test("valid --projection-screen-w and --projection-eye-w") {
    val options = MengerCLIOptions(Seq("--projection-screen-w", "1", "--projection-eye-w", "2"))
    assert(options.projectionScreenW() == 1)
    assert(options.projectionEyeW() == 2)
  }

  test("invalid --projection-screen-w (<=0)") {
    assertThrows[IllegalArgumentException]({
      MengerCLIOptions(Seq("--projection-screen-w", "0", "--projection-eye-w", "2"))
    })
  }

  test("invalid --projection-eye-w (<= --projection-screen-w)") {
    assertThrows[IllegalArgumentException]({
      MengerCLIOptions(Seq("--projection-screen-w", "2", "--projection-eye-w", "2"))
    })
  }

  test("valid --rot-x-w") {
    val options = MengerCLIOptions(Seq("--rot-x-w", "1"))
    assert(options.rotXW() == 1)
  }

  test("invalid --rot-x-w") {
    assertThrows[IllegalArgumentException]({
      MengerCLIOptions(Seq("--rot-x-w", "360"))
    })
  }
