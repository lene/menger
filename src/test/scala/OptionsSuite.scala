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

