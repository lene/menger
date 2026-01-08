package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ProfilingConfigSuite extends AnyFlatSpec with Matchers:

  "ProfilingConfig.disabled" should "have isEnabled = false" in:
    ProfilingConfig.disabled.isEnabled shouldBe false

  it should "have minDurationMs = None" in:
    ProfilingConfig.disabled.minDurationMs shouldBe None

  it should "have threshold = Int.MaxValue" in:
    ProfilingConfig.disabled.threshold shouldBe Int.MaxValue

  "ProfilingConfig.enabled" should "have isEnabled = true" in:
    ProfilingConfig.enabled(10).isEnabled shouldBe true

  it should "have correct minDurationMs" in:
    ProfilingConfig.enabled(10).minDurationMs shouldBe Some(10)
    ProfilingConfig.enabled(100).minDurationMs shouldBe Some(100)

  it should "have correct threshold" in:
    ProfilingConfig.enabled(10).threshold shouldBe 10
    ProfilingConfig.enabled(100).threshold shouldBe 100

  "ProfilingConfig case class" should "allow creating instances with None" in:
    val config = ProfilingConfig(None)
    config.isEnabled shouldBe false
    config.threshold shouldBe Int.MaxValue

  it should "allow creating instances with Some" in:
    val config = ProfilingConfig(Some(50))
    config.isEnabled shouldBe true
    config.threshold shouldBe 50
