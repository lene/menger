package menger.engines.scene

import menger.dsl.ResourceLimits

/** The triangles a scene may have, and what that number follows (menger#179): the smaller of
  * what the free GPU memory takes ([[VramBudget]]) and what the JVM heap takes
  * ([[HostHeapBudget]]), or the fixed `ResourceLimits.sceneMaxTriangles` where the GPU memory
  * cannot be read. Pure, so the engine's choice is testable without a device. */
object TriangleBudget:

  /** `triangles`; `origin` is what a refusal tells the user the number follows, empty for the
    * fixed default. */
  final case class Resolved(triangles: Long, origin: String)

  def resolve(gpu: Option[VramBudget.GpuMemory], maxHeapBytes: Long): Resolved =
    gpu.fold(Resolved(ResourceLimits.sceneMaxTriangles, "")) { memory =>
      val vram = VramBudget.triangleBudget(memory)
      val host = HostHeapBudget.triangleBudget(maxHeapBytes)
      if host < vram then
        Resolved(host, s"the triangle budget follows the JVM heap: ${mib(maxHeapBytes)} MiB " +
          "(raise it with -Xmx to render more triangles)")
      else
        Resolved(vram,
          s"the triangle budget follows the free GPU memory: ${mib(memory.freeBytes)} MiB free")
    }

  private def mib(bytes: Long): Long = bytes / VramBudget.BytesPerMiB
