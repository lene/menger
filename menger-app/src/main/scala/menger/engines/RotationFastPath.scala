package menger.engines

import io.github.lene.optix.MengerRenderer
import io.github.lene.optix.OptiXRenderer
import menger.Projection4DSpec
import menger.engines.scene.InstanceId

/** Strategy object for 4D-rotation fast paths in interactive and animation engines.
  *
  * Each fast path updates projection parameters on already-uploaded instances without
  * a full geometry rebuild. Two updaters remain: GPU-projected triangle meshes, and
  * the instanced IFS 4D types (menger4d / sierpinski4d / hexadecachoron4d), which now
  * share one projection-update call (F9).
  */
object RotationFastPath:

  /** Per-type projection update function signature. */
  type ProjectionUpdater = (OptiXRenderer, InstanceId, Projection4DSpec) => Unit

  /** Attempt a cached fast-path projection update for a set of specs.
    *
    * @param newSpecs      the current ObjectSpecs with possibly updated projection4D
    * @param renderer      the OptiX renderer instance
    * @param prevSpecs     the previously-cached specs (for diff detection)
    * @param prevIds       the previously-cached instance/slot ids per spec
    * @param updateFn      how to update one instance's projection on the renderer
    * @return true if the fast path was taken (projection-only delta), false otherwise
    */
  def tryFastPath(
    newSpecs: List[menger.ObjectSpec],
    renderer: OptiXRenderer,
    prevSpecs: List[menger.ObjectSpec],
    prevIds: Vector[Vector[InstanceId]],
    updateFn: ProjectionUpdater
  ): Boolean =
    if !WithAnimation.specsDifferOnlyIn4DProjection(prevSpecs, newSpecs) then false
    else
      require(prevSpecs.size == prevIds.size,
      s"prevSpecs.size (${prevSpecs.size}) != prevIds.size (${prevIds.size})")
    prevSpecs.lazyZip(newSpecs).lazyZip(prevIds).foreach {
        case (prevSpec, newSpec, ids) =>
          if prevSpec.projection4D != newSpec.projection4D then
            val proj = newSpec.projection4D.getOrElse(Projection4DSpec.default)
            ids.foreach { id => updateFn(renderer, id, proj) }
      }
      true

  /** Projection updater for GPU-projected 4D triangle meshes. */
  val gpuProjectionUpdater: ProjectionUpdater = (renderer, id, proj) =>
    renderer.updateMesh4DProjection(
      InstanceId.raw(id),
      eyeW = proj.eyeW, screenW = proj.screenW,
      rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW
    )

  /** Projection updater for all instanced-4D IFS types (menger4d / sierpinski4d /
    * hexadecachoron4d). The per-type distinction is add-time only, so a single
    * MengerRenderer.update4DProjection serves all three (F9). */
  val instanced4DUpdater: ProjectionUpdater = (renderer, id, proj) =>
    MengerRenderer.of(renderer).update4DProjection(
      InstanceId.raw(id),
      eyeW = proj.eyeW, screenW = proj.screenW,
      rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW
    )

  /** Attempt a GPU-projected 4D triangle mesh fast path using raw mesh slot indices.
    *
    * @param newSpecs      current ObjectSpecs
    * @param renderer      OptiX renderer
    * @param prevSpecs     previously-cached specs
    * @param prevSlots     previously-cached mesh slot indices per spec
    * @return true if projection-only delta was applied, false otherwise
    */
  def tryGpuProjectionFastPath(
    newSpecs: List[menger.ObjectSpec],
    renderer: OptiXRenderer,
    prevSpecs: List[menger.ObjectSpec],
    prevSlots: Vector[Vector[Int]]
  ): Boolean =
    if !WithAnimation.specsDifferOnlyIn4DProjection(prevSpecs, newSpecs) then false
    else
      prevSpecs.lazyZip(newSpecs).lazyZip(prevSlots).foreach {
        case (prevSpec, newSpec, slots) =>
          if prevSpec.projection4D != newSpec.projection4D then
            val proj = newSpec.projection4D.getOrElse(Projection4DSpec.default)
            slots.foreach { slot =>
              renderer.updateMesh4DProjection(
                slot,
                eyeW = proj.eyeW, screenW = proj.screenW,
                rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW
              )
            }
      }
      true
