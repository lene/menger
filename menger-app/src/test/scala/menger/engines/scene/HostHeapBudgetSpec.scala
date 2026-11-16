package menger.engines.scene

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// menger#179: the JVM heap bounds the scene too, not only the GPU memory.
class HostHeapBudgetSpec extends AnyFlatSpec with Matchers:

  import HostHeapBudget.*

  private val GiB = 1024L * 1024 * 1024

  "HostHeapBudget.triangleBudget" should "be the heap beyond the reserve over a triangle's cost" in:
    triangleBudget(8 * GiB) shouldBe (8 * GiB - ReserveBytes) / BytesPerTriangle

  it should "scale with the heap" in:
    triangleBudget(2 * GiB) should be < triangleBudget(8 * GiB)
    triangleBudget(8 * GiB) should be < triangleBudget(32 * GiB)

  it should "be zero, not negative, when the heap is smaller than the reserve" in:
    triangleBudget(ReserveBytes / 2) shouldBe 0L
    triangleBudget(0L) shouldBe 0L

  it should "hold one level-5 volume sponge in the packaged launcher's 8 GiB heap, but not two" in:
    // 13,062,144 triangles each (menger.objects.SpongeByVolume.triangleCount(5))
    triangleBudget(8 * GiB) should be >= 13_062_144L
    triangleBudget(8 * GiB) should be < 2 * 13_062_144L
