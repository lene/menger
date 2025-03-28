import menger.MengerCLIOptions
import menger.InteractiveMengerEngine
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MainSuite extends AnyFlatSpec with Matchers:

  "getConfig" should "return default config if no options" in :
    val options = MengerCLIOptions(Seq.empty)
    Main.getConfig(options)

  "createEngine" should "return default InteractiveMengerEngine if no options" in :
    val options = MengerCLIOptions(Seq.empty)
    Main.createEngine(options) shouldBe a [InteractiveMengerEngine]
