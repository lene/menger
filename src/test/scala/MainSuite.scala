import menger.MengerCLIOptions
import menger.engines.AnimatedMengerEngine
import menger.engines.InteractiveMengerEngine
import menger.engines.OptiXEngine
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// TODO: Replace asInstanceOf with pattern matching for type-safe casting
@SuppressWarnings(Array("org.wartremover.warts.AsInstanceOf"))
class MainSuite extends AnyFlatSpec with Matchers:

  "getConfig" should "return default config if no options" in :
    val options = MengerCLIOptions(Seq.empty)
    Main.getConfig(options)

  "createEngine" should "return default InteractiveMengerEngine if no options" in :
    val options = MengerCLIOptions(Seq.empty)
    Main.createEngine(options) shouldBe a [InteractiveMengerEngine]

  it should "return OptiXEngine when --optix flag is set" in:
    val options = MengerCLIOptions(Seq("--optix", "--sponge-type", "sphere"))
    Main.createEngine(options) shouldBe a [OptiXEngine]

  it should "pass correct radius to OptiXEngine" in:
    val options = MengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--radius", "2.5"))
    val engine = Main.createEngine(options)
    engine shouldBe a [OptiXEngine]
    engine.asInstanceOf[OptiXEngine].sphereRadius shouldEqual 2.5f

  it should "pass default radius (1.0) to OptiXEngine when not specified" in:
    val options = MengerCLIOptions(Seq("--optix", "--sponge-type", "sphere"))
    val engine = Main.createEngine(options)
    engine shouldBe a [OptiXEngine]
    engine.asInstanceOf[OptiXEngine].sphereRadius shouldEqual 1.0f

  it should "return InteractiveMengerEngine when --optix is false" in:
    val options = MengerCLIOptions(Seq("--sponge-type", "cube"))
    Main.createEngine(options) shouldBe a [InteractiveMengerEngine]

  it should "return AnimatedMengerEngine when --animate is specified" in:
    val options = MengerCLIOptions(Seq("--animate", "frames=10:rot-y=0-360"))
    Main.createEngine(options) shouldBe a [AnimatedMengerEngine]
