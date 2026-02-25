package menger.cli

import org.rogach.scallop.exceptions.ScallopException
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TAnimationCLIOptionsSuite extends AnyFlatSpec with Matchers:

  "--t" should "accept a float value with --optix --scene" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--scene", "glass-sphere", "--t", "0.5"))
    opts.freezeT() shouldBe 0.5f

  it should "reject without --scene" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--objects", "type=sphere", "--t", "0.5"))

  it should "reject without --optix" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--scene", "glass-sphere", "--t", "0.5"))

  it should "be mutually exclusive with --start-t" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--scene", "glass-sphere", "--t", "0.5", "--start-t", "0"))

  it should "be mutually exclusive with --end-t" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--scene", "glass-sphere", "--t", "0.5", "--end-t", "1"))

  it should "be mutually exclusive with --frames" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq(
        "--optix", "--scene", "glass-sphere", "--t", "0.5",
        "--frames", "10", "--save-name", "f_%04d.png"
      ))

  it should "be mutually exclusive with --animate" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq(
        "--optix", "--scene", "glass-sphere", "--t", "0.5",
        "--animate", "frames=10:rot-y=0-360"
      ))

  "--frames" should "accept a positive integer" in:
    val opts = SafeMengerCLIOptions(Seq(
      "--optix", "--scene", "glass-sphere",
      "--frames", "100", "--save-name", "orbit_%04d.png"
    ))
    opts.tFrames() shouldBe 100

  it should "reject zero" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq(
        "--optix", "--scene", "glass-sphere",
        "--frames", "0", "--save-name", "f_%04d.png"
      ))

  it should "reject negative values" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq(
        "--optix", "--scene", "glass-sphere",
        "--frames", "-5", "--save-name", "f_%04d.png"
      ))

  it should "require --scene" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq(
        "--optix", "--objects", "type=sphere",
        "--frames", "10", "--save-name", "f_%04d.png"
      ))

  it should "require --optix" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq(
        "--scene", "glass-sphere",
        "--frames", "10", "--save-name", "f_%04d.png"
      ))

  it should "require --save-name with %" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq(
        "--optix", "--scene", "glass-sphere",
        "--frames", "10", "--save-name", "output.png"
      ))

  it should "require --save-name" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq(
        "--optix", "--scene", "glass-sphere",
        "--frames", "10"
      ))

  it should "be mutually exclusive with --animate" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq(
        "--optix", "--scene", "glass-sphere",
        "--frames", "10", "--save-name", "f_%04d.png",
        "--animate", "frames=10:rot-y=0-360"
      ))

  "--start-t and --end-t" should "have defaults of 0 and 1" in:
    val opts = SafeMengerCLIOptions(Seq(
      "--optix", "--scene", "glass-sphere",
      "--frames", "10", "--save-name", "f_%04d.png"
    ))
    opts.startT() shouldBe 0f
    opts.endT() shouldBe 1f

  it should "accept custom values" in:
    val opts = SafeMengerCLIOptions(Seq(
      "--optix", "--scene", "glass-sphere",
      "--frames", "100", "--start-t", "0", "--end-t", "6.28",
      "--save-name", "orbit_%04d.png"
    ))
    opts.startT() shouldBe 0f
    opts.endT() shouldBe 6.28f

  "all t-animation options" should "work together for multi-frame animation" in:
    val opts = SafeMengerCLIOptions(Seq(
      "--optix", "--scene", "glass-sphere",
      "--frames", "100", "--start-t", "0", "--end-t", "6.28",
      "--save-name", "orbit_%04d.png", "--headless"
    ))
    opts.tFrames() shouldBe 100
    opts.startT() shouldBe 0f
    opts.endT() shouldBe 6.28f
    opts.saveName() shouldBe "orbit_%04d.png"
    opts.headless() shouldBe true
