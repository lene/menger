package menger.engines

import io.github.lene.optix.MeshSlot
import io.github.lene.optix.OptiXRenderer
import menger.engines.scene.InstanceId
import menger.geometry.MengerRenderer
import menger.spec.ObjectSpec
import menger.spec.Projection4DSpec

/** Strategy object for 4D-rotation fast paths in interactive and animation engines.
  *
  * Each fast path updates projection parameters on already-uploaded instances without
  * a full geometry rebuild. Two updaters remain: GPU-projected triangle meshes (by mesh slot),
  * and the instanced IFS 4D types (menger4d / sierpinski4d / hexadecachoron4d, by instance),
  * which share one projection-update call (F9).
  *
  * One decision for all of them (review 2026-10-14, menger#102): there were three copies, and
  * in one the guard sat outside its `else`, so it updated every instance and returned true for
  * any change -- harmless only because its one caller pre-filtered.
  */
object RotationFastPath:

  /** How to update one instance's (or mesh slot's) projection on the renderer. */
  type ProjectionUpdater[Id] = (OptiXRenderer, Id, Projection4DSpec) => Unit

  /** Updates the projection of every instance whose spec's projection changed, and returns true,
    * iff `newSpecs` differ from `prevSpecs` at most in their 4D projection and there is one id
    * list per spec. Otherwise changes nothing and returns false: the caller rebuilds the scene.
    *
    * @param prevIds  the instances (or mesh slots) each previous spec was built as
    */
  def tryFastPath[Id](
    newSpecs: List[ObjectSpec],
    renderer: OptiXRenderer,
    prevSpecs: List[ObjectSpec],
    prevIds: Vector[Vector[Id]],
    update: ProjectionUpdater[Id]
  ): Boolean =
    val eligible = prevIds.size == prevSpecs.size &&
      WithAnimation.specsDifferAtMostIn4DProjection(prevSpecs, newSpecs)
    if eligible then
      prevSpecs.lazyZip(newSpecs).lazyZip(prevIds).foreach { (prevSpec, newSpec, ids) =>
        if prevSpec.projection4D != newSpec.projection4D then
          val proj = newSpec.projection4D.getOrElse(Projection4DSpec.default)
          ids.foreach(id => update(renderer, id, proj))
      }
    eligible

  /** GPU-projected 4D triangle meshes, by mesh slot. */
  val meshSlotUpdater: ProjectionUpdater[MeshSlot] = (renderer, slot, proj) =>
    renderer.updateMesh4DProjection(
      slot,
      eyeW = proj.eyeW, screenW = proj.screenW,
      rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW
    )

  /** All instanced-4D IFS types (menger4d / sierpinski4d / hexadecachoron4d). The per-type
    * distinction is add-time only, so a single MengerRenderer.update4DProjection serves all
    * three (F9). */
  val instanced4DUpdater: ProjectionUpdater[InstanceId] = (renderer, id, proj) =>
    MengerRenderer.of(renderer).update4DProjection(
      id,
      eyeW = proj.eyeW, screenW = proj.screenW,
      rotXW = proj.rotXW, rotYW = proj.rotYW, rotZW = proj.rotZW
    )
