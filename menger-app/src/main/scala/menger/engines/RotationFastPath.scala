package menger.engines

import menger.Projection4DSpec
import menger.engines.scene.InstanceId
import io.github.lene.optix.OptiXRenderer

/** Strategy object for 4D-rotation fast paths in interactive and animation engines.
  *
  * Each fast path updates projection parameters on already-uploaded instances without
  * a full geometry rebuild. The 5 variants differ only in the renderer projection-
  * update method and the cache state type — this object consolidates the copy-paste.
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

  /** Projection updater for Menger4D IFS instances. */
  val menger4DUpdater: ProjectionUpdater = (renderer, id, proj) =>
    renderer.updateMenger4DProjection(
      InstanceId.raw(id),
      eyeW = proj.eyeW, screenW = proj.screenW,
      rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW
    )

  /** Projection updater for Sierpinski4D IFS instances. */
  val sierpinski4DUpdater: ProjectionUpdater = (renderer, id, proj) =>
    renderer.updateSierpinski4DProjection(
      InstanceId.raw(id),
      eyeW = proj.eyeW, screenW = proj.screenW,
      rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW
    )

  /** Projection updater for Hexadecachoron4D IFS instances. */
  val hexadecachoron4DUpdater: ProjectionUpdater = (renderer, id, proj) =>
    renderer.updateHexadecachoron4DProjection(
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
