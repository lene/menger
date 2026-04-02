import menger.MengerCLIOptions
import menger.engines.InteractiveEngine
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MainSuite extends AnyFlatSpec with Matchers:

  "getConfig" should "return default config if no options" in :
    val options = MengerCLIOptions(Seq.empty)
    Main.getConfig(options)

  "createEngine" should "return InteractiveEngine when --objects is set" in:
    val options = MengerCLIOptions(Seq("--objects", "type=sphere"))
    Main.createEngine(options) shouldBe a [InteractiveEngine]

  it should "return InteractiveEngine when --objects specifies a cube" in:
    val options = MengerCLIOptions(Seq("--objects", "type=cube"))
    Main.createEngine(options) shouldBe a [InteractiveEngine]
