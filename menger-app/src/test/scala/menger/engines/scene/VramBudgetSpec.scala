package menger.engines.scene

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// menger#179: the scene's triangle budget follows the free VRAM instead of a fixed count.
class VramBudgetSpec extends AnyFlatSpec with Matchers:

  import VramBudget.*

  private val GiB = 1024L * 1024 * 1024

  "VramBudget.triangleBudget" should "be the free bytes beyond the headroom over a triangle's cost" in:
    triangleBudget(GpuMemory(freeBytes = 4 * GiB)) shouldBe
      (4 * GiB - HeadroomBytes) / BytesPerTriangle

  it should "scale with the free memory: a bigger GPU accepts more, a smaller one fewer" in:
    val small = triangleBudget(GpuMemory(freeBytes = 2 * GiB))
    val big = triangleBudget(GpuMemory(freeBytes = 4 * GiB))
    val huge = triangleBudget(GpuMemory(freeBytes = 24 * GiB))
    small should be < big
    big should be < huge
    huge should be > 100_000_000L

  it should "reserve what the first render allocates while it has not happened yet" in:
    val before = triangleBudget(GpuMemory(freeBytes = 6 * GiB, firstRenderPending = true))
    val after = triangleBudget(GpuMemory(freeBytes = 6 * GiB, firstRenderPending = false))
    (after - before) * BytesPerTriangle should be >= FirstRenderReserveBytes - BytesPerTriangle

  it should "count the triangles of the scene about to be replaced as free" in:
    val resident = 10_000_000L
    val free = 3 * GiB
    triangleBudget(GpuMemory(free, residentTriangles = resident)) shouldBe
      triangleBudget(GpuMemory(free + resident * BytesPerTriangle))

  it should "be zero, not negative, when less than the headroom is free" in:
    triangleBudget(GpuMemory(freeBytes = HeadroomBytes / 2)) shouldBe 0L
    triangleBudget(GpuMemory(freeBytes = 0L, firstRenderPending = true)) shouldBe 0L

  it should "be roughly today's fixed budget on the 6 GB development GPU after the first render" in:
    // measured 2026-11-16: 4103 MiB free after the first render of an empty scene
    val budget = triangleBudget(GpuMemory(freeBytes = 4103L * 1024 * 1024))
    budget should be > 16_000_000L
    budget should be < 30_000_000L
