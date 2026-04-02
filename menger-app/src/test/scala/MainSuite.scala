import menger.MengerCLIOptions
import menger.engines.OptiXEngine
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MainSuite extends AnyFlatSpec with Matchers:

  "getConfig" should "return default config if no options" in :
    val options = MengerCLIOptions(Seq.empty)
    Main.getConfig(options)

  "createEngine" should "return OptiXEngine when --objects is set" in:
    val options = MengerCLIOptions(Seq("--objects", "type=sphere"))
    Main.createEngine(options) shouldBe a [OptiXEngine]

  it should "return OptiXEngine when --objects specifies a cube" in:
    val options = MengerCLIOptions(Seq("--objects", "type=cube"))
    Main.createEngine(options) shouldBe a [OptiXEngine]
