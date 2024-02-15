import org.rogach.scallop.exceptions.ScallopException
import org.scalatest.funsuite.AnyFunSuite

class OptionsSuite extends AnyFunSuite:
  test("empty options give default timeout") {
    val options = MengerCLIOptions(Seq[String]())
    assert(options.timeout() == 0)
  }

  test("--timeout option sets timeout") {
    val options = MengerCLIOptions(Seq[String]("--timeout", "1"))
    assert(options.timeout() == 1)
  }

  test("--sponge-type box|square") {
    Seq("box", "square").foreach { spongeType =>
      val options = MengerCLIOptions(Seq[String]("--sponge-type", spongeType))
      assert(options.spongeType() == spongeType)
    }
  }

  test("default for --sponge-type") {
    val options = MengerCLIOptions(Seq[String]())
    assert(options.spongeType() == "box")
  }


  test("invalid --sponge-type throws IllegalArgumentException") {
    assertThrows[IllegalArgumentException]({
      MengerCLIOptions(Seq[String]("--sponge-type", "invalid"))
    })
  }

  test("--antialias-samples") {
    val options = MengerCLIOptions(Seq[String]("--antialias-samples", "1"))
    assert(options.antialiasSamples() == 1)
  }

  test("getConfig") {

    val options = MengerCLIOptions(Seq[String]())
    val config = Main.getConfig(options)
  }