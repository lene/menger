package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BaseEngineMaxInstancesSuite extends AnyFlatSpec with Matchers:

  "BaseEngine.adjustedMaxInstances" should "keep the budget when the scene fits" in:
    BaseEngine.adjustedMaxInstances(required = 10, configured = 64) shouldBe 64

  it should "keep the budget when the requirement is unknown" in:
    BaseEngine.adjustedMaxInstances(required = 0, configured = 64) shouldBe 64

  it should "double the requirement when the scene does not fit" in:
    BaseEngine.adjustedMaxInstances(required = 10849, configured = 64) shouldBe 21698

  it should "cap the budget at the global limit" in:
    BaseEngine.adjustedMaxInstances(
      required = menger.common.Const.maxInstancesLimit,
      configured = 64
    ) shouldBe menger.common.Const.maxInstancesLimit
