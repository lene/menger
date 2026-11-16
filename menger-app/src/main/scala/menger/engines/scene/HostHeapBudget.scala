package menger.engines.scene

/** How many mesh triangles the JVM heap takes while a scene is built (menger#179). The fixed
  * 16M budget that the free GPU memory replaced bounded this too: a scene that fits the GPU can
  * still run the host out of memory, which is a crash, not a refusal. The budget the engine
  * uses is the smaller of this and [[VramBudget]]'s.
  *
  * Measured 2026-11-16 as the smallest `-Xmx` that builds the mesh (bisected, 16 MiB steps; the
  * mesh builders in a JVM without the renderer, which is where the heap goes before the upload,
  * the arrays being copied to native memory by JNI):
  *  - `sponge-volume` level 4 (672,768 triangles): 79 MiB; level 5 (13,062,144): 1,230 MiB, 99 B
  *    per triangle (76 B of the finished arrays, the rest while it is built).
  *  - `sponge-surface` level 4 (248,832 triangles): 142 MiB, 510 B per triangle with the JVM's
  *    own 15 MiB; level 5 (3.1M) did not finish in 170 s at 6 GiB and held 1.4 GiB at its last
  *    collection: the surface builder keeps its subdivision's intermediate faces.
  *  - the old fixed budget came to the same: 16M triangles in the packaged launcher's 8 GiB
  *    heap is 512 B per triangle, and the heap one level-5 volume sponge used in the app was
  *    measured at about 4.5 GB when the heap was not constrained (2026-10-13, ResourceLimits).
  */
object HostHeapBudget:

  /** The heap one triangle takes at the peak of building its mesh: the surface sponge's 510 B,
    * the most measured, rounded to a power of two. The volume sponge takes a fifth of that, so
    * a scene of volume sponges is judged about five times too dear, which only matters on a
    * heap too small for it: raise `-Xmx`, the refusal says so. */
  val BytesPerTriangle = 512L

  /** Kept free for what the scene does not own: the JVM's own classes (15 MiB measured), libGDX,
    * the DSL scene and the builders' working data. */
  val ReserveBytes = 256L * 1024 * 1024

  /** The most triangles a scene can have with `maxHeapBytes` of heap (`Runtime.maxMemory`);
    * zero when not even the reserve fits. */
  def triangleBudget(maxHeapBytes: Long): Long =
    math.max(0L, (maxHeapBytes - ReserveBytes) / BytesPerTriangle)
