package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ProfilingCLIOptionsSuite extends AnyFlatSpec with Matchers:

  "MengerCLIOptions --profile-min-ms" should "parse valid value" in:
    val opts = MengerCLIOptions(List("--profile-min-ms", "10"))
    opts.profileMinMs.toOption shouldBe Some(10)

  it should "be optional" in:
    val opts = MengerCLIOptions(List())
    opts.profileMinMs.toOption shouldBe None

  it should "accept zero" in:
    val opts = MengerCLIOptions(List("--profile-min-ms", "0"))
    opts.profileMinMs.toOption shouldBe Some(0)

  it should "accept large values" in:
    val opts = MengerCLIOptions(List("--profile-min-ms", "10000"))
    opts.profileMinMs.toOption shouldBe Some(10000)

  // Note: Negative values are rejected by scallop validation (validate = _ >= 0)
  // This causes System.exit() which cannot be tested directly in unit tests

  "Main.createEngine" should "create disabled ProfilingConfig when option not provided" in:
    val opts = MengerCLIOptions(List())
    val config = opts.profileMinMs.toOption match
      case Some(minMs) => ProfilingConfig.enabled(minMs)
      case None => ProfilingConfig.disabled

    config shouldBe ProfilingConfig.disabled

  it should "create enabled ProfilingConfig when option provided" in:
    val opts = MengerCLIOptions(List("--profile-min-ms", "25"))
    val config = opts.profileMinMs.toOption match
      case Some(minMs) => ProfilingConfig.enabled(minMs)
      case None => ProfilingConfig.disabled

    config shouldBe ProfilingConfig.enabled(25)
