import menger.MengerCLIOptions
import menger.engines.AnimatedMengerEngine
import menger.engines.InteractiveMengerEngine
import menger.engines.OptiXEngine
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MainSuite extends AnyFlatSpec with Matchers:

  "getConfig" should "return default config if no options" in :
    val options = MengerCLIOptions(Seq.empty)
    Main.getConfig(options)

  "createEngine" should "return default InteractiveMengerEngine if no options" in :
    val options = MengerCLIOptions(Seq.empty)
    Main.createEngine(options) shouldBe a [InteractiveMengerEngine]

  it should "return OptiXEngine when --optix flag is set" in:
    val options = MengerCLIOptions(Seq("--optix", "--objects", "type=sphere"))
    Main.createEngine(options) shouldBe a [OptiXEngine]

  it should "return InteractiveMengerEngine when --optix is false" in:
    val options = MengerCLIOptions(Seq("--sponge-type", "cube"))
    Main.createEngine(options) shouldBe a [InteractiveMengerEngine]

  it should "return AnimatedMengerEngine when --animate is specified" in:
    val options = MengerCLIOptions(Seq("--animate", "frames=10:rot-y=0-360"))
    Main.createEngine(options) shouldBe a [AnimatedMengerEngine]
