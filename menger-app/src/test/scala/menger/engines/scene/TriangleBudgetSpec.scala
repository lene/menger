package menger.engines.scene

import menger.dsl.ResourceLimits
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// menger#179: the smaller of what the GPU memory takes and what the JVM heap takes, the fixed
// budget where the GPU memory is unknown.
class TriangleBudgetSpec extends AnyFlatSpec with Matchers:

  private val GiB = 1024L * 1024 * 1024
  private def gpu(free: Long) = Some(VramBudget.GpuMemory(free))

  "TriangleBudget.resolve" should "be the fixed budget, naming no origin, without GPU memory" in:
    TriangleBudget.resolve(None, maxHeapBytes = 64 * GiB) shouldBe
      TriangleBudget.Resolved(ResourceLimits.sceneMaxTriangles, "")

  it should "follow the GPU memory when that is the smaller bound" in:
    val resolved = TriangleBudget.resolve(gpu(2 * GiB), maxHeapBytes = 64 * GiB)
    resolved.triangles shouldBe VramBudget.triangleBudget(VramBudget.GpuMemory(2 * GiB))
    resolved.origin should include("free GPU memory: 2048 MiB free")

  it should "follow the JVM heap when that is the smaller bound" in:
    val resolved = TriangleBudget.resolve(gpu(24 * GiB), maxHeapBytes = 4 * GiB)
    resolved.triangles shouldBe HostHeapBudget.triangleBudget(4 * GiB)
    resolved.origin should include("JVM heap: 4096 MiB")
    resolved.origin should include("-Xmx")

  it should "never exceed either bound" in:
    for
      free <- List(1 * GiB, 4 * GiB, 24 * GiB)
      heap <- List(2 * GiB, 8 * GiB, 64 * GiB)
    do
      TriangleBudget.resolve(gpu(free), heap).triangles shouldBe
        math.min(VramBudget.triangleBudget(VramBudget.GpuMemory(free)),
          HostHeapBudget.triangleBudget(heap))
