package menger.engines.scene

/** How many mesh triangles the GPU's free memory takes (menger#179), replacing the fixed budget
  * `ResourceLimits.sceneMaxTriangles` wherever the renderer can say how much is free. Pure
  * arithmetic over what the engine reads from the device, so the limit follows the card: a GPU
  * with more free memory accepts bigger scenes, one with less refuses them cleanly before the
  * upload runs out of memory.
  *
  * Measured on the RTX A1000 (6 GB), 2026-11-16, with 0.25M to 30M triangles of quads with
  * unshared corners and with the level 4 and 5 volume sponges (OptiX 9, optix-jni 0.5.4):
  *  - a triangle costs 77 B of vertex and index buffers (8 floats per vertex, two vertices per
  *    triangle, 3 indices), 57 B of compacted acceleration structure (134 B settled), and about 7 B
  *    more while the structure is built (build scratch): 141 B at the peak, flat from 0.25M to
  *    30M triangles, the same for the sponges (134 B settled, 141-143 B at the peak). With 4,103
  *    MiB free, the upload of 30.0M triangles succeeded and of 31.5M failed with a CUDA out of
  *    memory error, 30.5M predicted by 141 B per triangle.
  *  - an empty scene costs 107 MiB at `initialize` and 1,586 MiB more at the first render (the
  *    caustics hit-point and grid buffers, fixed in size); 8 MiB more of frame buffers at 1080p
  *    and 32 MiB at 4K. The denoiser adds about 420 MiB at 1080p (about 1.9 GiB at 4K, not
  *    modelled: a 4K denoised render of a scene near the budget can run out of memory).
  */
object VramBudget:

  val BytesPerMiB = 1024L * 1024

  /** The device memory one triangle takes through upload, acceleration structure build and
    * compaction: measured 141 B at the peak for 8-float vertices, 8 B more for the 9-float
    * vertices with per-vertex alpha, rounded up (7 %) for meshes that share fewer vertices than
    * the sponges' quads. */
  val BytesPerTriangle = 160L

  /** Kept free for everything the measurements do not model: other programs' memory
    * fluctuating, frame buffers at a larger size, instance and shader binding tables. */
  val HeadroomBytes = 256 * BytesPerMiB

  /** What the first render allocates after `initialize` and a scene upload: the caustics buffers
    * (1,586 MiB measured, independent of the scene) and the denoiser's state at 1080p (420 MiB),
    * rounded up. Reserved only before that render, afterwards the free memory excludes it. */
  val FirstRenderReserveBytes = 2304 * BytesPerMiB

  /** What the engine knows about the device when it checks a scene.
    *
    * @param freeBytes          `cudaMemGetInfo`'s free device memory
    * @param residentTriangles  triangles of the scene the build is about to replace: a rebuild
    *                           clears it first, so its memory counts as free
    * @param firstRenderPending whether the renderer has yet to make its first render, which
    *                           allocates [[FirstRenderReserveBytes]] on top of the scene */
  final case class GpuMemory(
    freeBytes: Long,
    residentTriangles: Long = 0L,
    firstRenderPending: Boolean = false
  )

  /** The most triangles a scene can have on a device in this state; zero when not even the
    * headroom is free. */
  def triangleBudget(memory: GpuMemory): Long =
    val reusable = memory.freeBytes + memory.residentTriangles * BytesPerTriangle
    val firstRender = if memory.firstRenderPending then FirstRenderReserveBytes else 0L
    val reserved = HeadroomBytes + firstRender
    math.max(0L, (reusable - reserved) / BytesPerTriangle)
