package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.rogach.scallop.exceptions.ScallopException

class CausticsCLIOptionsSuite extends AnyFlatSpec with Matchers:
  class SafeMengerCLIOptions(args: Seq[String]) extends menger.MengerCLIOptions(args):
    @SuppressWarnings(Array("org.wartremover.warts.Throw"))
    override def onError(e: Throwable): Unit = throw e

  "--caustics" should "default to false when not provided" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere"))
    opts.caustics() shouldBe false

  it should "be enabled when flag is provided" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics"))
    opts.caustics() shouldBe true

  it should "reject --caustics without --optix flag" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--caustics"))

  "--caustics-photons" should "default to 100000" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics"))
    opts.causticsPhotons() shouldBe 100000

  it should "accept custom photon count" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-photons", "50000"))
    opts.causticsPhotons() shouldBe 50000

  it should "accept maximum photon count (10 million)" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-photons", "10000000"))
    opts.causticsPhotons() shouldBe 10000000

  it should "reject photon count exceeding 10 million" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-photons", "10000001"))

  it should "reject zero photon count" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-photons", "0"))

  it should "reject negative photon count" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-photons", "-1"))

  it should "require --caustics flag" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics-photons", "50000"))

  "--caustics-iterations" should "default to 10" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics"))
    opts.causticsIterations() shouldBe 10

  it should "accept custom iteration count" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-iterations", "50"))
    opts.causticsIterations() shouldBe 50

  it should "accept maximum iteration count (1000)" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-iterations", "1000"))
    opts.causticsIterations() shouldBe 1000

  it should "reject iteration count exceeding 1000" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-iterations", "1001"))

  it should "reject zero iterations" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-iterations", "0"))

  it should "require --caustics flag" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics-iterations", "20"))

  "--caustics-radius" should "default to 0.1" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics"))
    opts.causticsRadius() shouldBe 0.1f

  it should "accept custom radius" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-radius", "0.5"))
    opts.causticsRadius() shouldBe 0.5f

  it should "accept maximum radius (10.0)" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-radius", "10.0"))
    opts.causticsRadius() shouldBe 10.0f

  it should "reject radius exceeding 10.0" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-radius", "10.1"))

  it should "reject zero radius" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-radius", "0.0"))

  it should "reject negative radius" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-radius", "-0.5"))

  it should "require --caustics flag" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics-radius", "0.2"))

  "--caustics-alpha" should "default to 0.7" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics"))
    opts.causticsAlpha() shouldBe 0.7f

  it should "accept custom alpha value" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-alpha", "0.5"))
    opts.causticsAlpha() shouldBe 0.5f

  it should "reject alpha >= 1.0" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-alpha", "1.0"))

  it should "reject alpha <= 0.0" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-alpha", "0.0"))

  it should "reject negative alpha" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics", "--caustics-alpha", "-0.5"))

  it should "require --caustics flag" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--caustics-alpha", "0.8"))

  "all caustics options" should "work together" in:
    val opts = SafeMengerCLIOptions(Seq(
      "--optix", "--sponge-type", "sphere", "--caustics",
      "--caustics-photons", "200000",
      "--caustics-iterations", "25",
      "--caustics-radius", "0.2",
      "--caustics-alpha", "0.6"
    ))
    opts.caustics() shouldBe true
    opts.causticsPhotons() shouldBe 200000
    opts.causticsIterations() shouldBe 25
    opts.causticsRadius() shouldBe 0.2f
    opts.causticsAlpha() shouldBe 0.6f
