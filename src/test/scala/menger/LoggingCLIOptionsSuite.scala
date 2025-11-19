package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.rogach.scallop.exceptions.ScallopException

class LoggingCLIOptionsSuite extends AnyFlatSpec with Matchers:
  
  class SafeMengerCLIOptions(args: Seq[String]) extends menger.MengerCLIOptions(args):
    @SuppressWarnings(Array("org.wartremover.warts.Throw"))
    override def onError(e: Throwable): Unit = throw e

  "MengerCLIOptions --log-level" should "default to INFO" in:
    val opts = SafeMengerCLIOptions(List())
    opts.logLevel() shouldBe "INFO"

  it should "accept ERROR" in:
    val opts = SafeMengerCLIOptions(List("--log-level", "ERROR"))
    opts.logLevel() shouldBe "ERROR"

  it should "accept WARN" in:
    val opts = SafeMengerCLIOptions(List("--log-level", "WARN"))
    opts.logLevel() shouldBe "WARN"

  it should "accept INFO" in:
    val opts = SafeMengerCLIOptions(List("--log-level", "INFO"))
    opts.logLevel() shouldBe "INFO"

  it should "accept DEBUG" in:
    val opts = SafeMengerCLIOptions(List("--log-level", "DEBUG"))
    opts.logLevel() shouldBe "DEBUG"

  it should "accept TRACE" in:
    val opts = SafeMengerCLIOptions(List("--log-level", "TRACE"))
    opts.logLevel() shouldBe "TRACE"

  it should "accept lowercase error" in:
    val opts = SafeMengerCLIOptions(List("--log-level", "error"))
    opts.logLevel() shouldBe "error"

  it should "accept mixed case WaRn" in:
    val opts = SafeMengerCLIOptions(List("--log-level", "WaRn"))
    opts.logLevel() shouldBe "WaRn"

  it should "reject invalid log level" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(List("--log-level", "INVALID"))

  it should "reject empty string" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(List("--log-level", ""))

  it should "reject ALL (not a standard SLF4J level)" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(List("--log-level", "ALL"))

  it should "reject OFF (not a standard SLF4J level)" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(List("--log-level", "OFF"))

  "MengerCLIOptions --fps-log-interval" should "default to 1000" in:
    val opts = SafeMengerCLIOptions(List())
    opts.fpsLogInterval() shouldBe 1000

  it should "accept positive values" in:
    val opts = SafeMengerCLIOptions(List("--fps-log-interval", "500"))
    opts.fpsLogInterval() shouldBe 500

  it should "accept 1 millisecond" in:
    val opts = SafeMengerCLIOptions(List("--fps-log-interval", "1"))
    opts.fpsLogInterval() shouldBe 1

  it should "accept large values" in:
    val opts = SafeMengerCLIOptions(List("--fps-log-interval", "60000"))
    opts.fpsLogInterval() shouldBe 60000

  it should "reject zero" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(List("--fps-log-interval", "0"))

  it should "reject negative values" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(List("--fps-log-interval", "-100"))
