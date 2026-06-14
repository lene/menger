package menger.engines

import java.util.concurrent.atomic.AtomicInteger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class VideoFrameCacheSuite extends AnyFlatSpec with Matchers:

  "VideoFrameCache" should "reuse a decoded frame for the same timestamp" in:
    val cache = VideoFrameCache(maxFrames = 2)
    val decodes = AtomicInteger(0)

    val first = cache.getOrDecode(0.5):
      decodes.incrementAndGet()
      Array[Byte](1)
    val second = cache.getOrDecode(0.5):
      decodes.incrementAndGet()
      Array[Byte](2)

    first shouldBe Array[Byte](1)
    second shouldBe first
    decodes.get shouldBe 1
    cache.size shouldBe 1

  it should "evict least-recently-used frames beyond the configured maximum" in:
    val cache = VideoFrameCache(maxFrames = 2)

    cache.getOrDecode(0.0)(Array[Byte](0))
    cache.getOrDecode(1.0)(Array[Byte](1))
    cache.getOrDecode(0.0)(Array[Byte](9))
    cache.getOrDecode(2.0)(Array[Byte](2))

    cache.size shouldBe 2
    cache.contains(0.0) shouldBe true
    cache.contains(1.0) shouldBe false
    cache.contains(2.0) shouldBe true

  it should "reject a non-positive maximum" in:
    an[IllegalArgumentException] should be thrownBy VideoFrameCache(maxFrames = 0)
