package menger.engines.scene

import scala.collection.mutable
import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.ObjectRotation
import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.ProfilingConfig

/** A 4D triangle-mesh scene built with its renderer handles recorded, so that an animation can
  * update it in place instead of rebuilding it every frame (usability review 2026-09, session 2,
  * F52: the preview window rebuilt a level-2.x tesseract sponge every frame, ~20 s each, and
  * skipped most frames). For every level in [n, n+1) the geometry is the same, level n+1 plus
  * level n's hole caps; only the 4D projection and the caps' alpha change. Position and 3D
  * rotation are instance transforms, changed without touching the geometry. */
object TrackedMesh4D:

  /** Per spec: its projected-mesh slots, all its instances and, for a fractional sponge, its
    * hole-cap instance. */
  final case class State(
    specs: List[ObjectSpec],
    slotsPerSpec: Vector[Vector[Int]],
    instancesPerSpec: Vector[Vector[Int]],
    capsInstancePerSpec: Vector[Option[Int]]
  )

  /** True when `next` can be reached from `prev` by `updateInPlace`: the same objects, differing
    * at most in their 4D projection, the fractional part of their level, their position and
    * their 3D rotation. */
  def canUpdateInPlace(prev: List[ObjectSpec], next: List[ObjectSpec]): Boolean =
    prev.length == next.length && prev.lazyZip(next).forall((a, b) => geometry(a) == geometry(b))

  // The view drops out, the w-scale stays: it changes the 4D mesh itself (menger#65).
  private def geometry(spec: ObjectSpec): ObjectSpec =
    spec.withoutView.copy(
      level = spec.level.map(levelInterval), x = 0f, y = 0f, z = 0f, rotation = ObjectRotation()
    )

  private def pose(spec: ObjectSpec) = (spec.x, spec.y, spec.z, spec.rotation)

  // Every fractional level in [n, n+1) builds the same meshes; an integer level builds one.
  private val FractionalMarker = 0.5f
  private def levelInterval(level: Float): Float =
    if level == level.floor then level else level.floor + FractionalMarker

  /** Builds `specs` with a recording TriangleMeshSceneBuilder; `None` when some spec produced
    * no projected mesh, so the scene cannot be updated in place. */
  def build(
    specs: List[ObjectSpec],
    renderer: OptiXRenderer,
    textureDir: String,
    maxInstances: SceneBuilder => Int
  )(using ProfilingConfig): Try[Option[State]] =
    val slots = mutable.Map.empty[Int, mutable.ArrayBuffer[Int]]
    val instances = mutable.Map.empty[Int, mutable.ArrayBuffer[Int]]
    val caps = mutable.Map.empty[Int, Int]
    val builder = TriangleMeshSceneBuilder(
      textureDir,
      mesh4DRecorder = (spec, slot) => slots.getOrElseUpdate(spec, mutable.ArrayBuffer.empty) += slot,
      holeCapsRecorder = (spec, instance) => caps(spec) = instance,
      instanceRecorder =
        (spec, instance) => instances.getOrElseUpdate(spec, mutable.ArrayBuffer.empty) += instance
    )
    builder.validateAndBuild(specs, renderer, maxInstances(builder)).map { _ =>
      Option.when(slots.size == specs.size)(State(
        specs,
        specs.indices.map(i => slots(i).toVector).toVector,
        specs.indices.map(i => instances.get(i).fold(Vector.empty)(_.toVector)).toVector,
        specs.indices.map(caps.get).toVector
      ))
    }

  /** Applies `next` (for which `canUpdateInPlace(state.specs, next)` holds) to the built scene:
    * the new projection on every projected mesh, the new hole-cap alpha on every caps instance,
    * the new position/rotation on every instance (an IAS-only rebuild on the next render). */
  def updateInPlace(state: State, next: List[ObjectSpec], renderer: OptiXRenderer): State =
    next.indices.foreach { i =>
      val (prev, spec) = (state.specs(i), next(i))
      if pose(prev) != pose(spec) then
        val transform = TriangleMeshSceneBuilder.instanceTransform(spec)
        state.instancesPerSpec(i).foreach(id => renderer.setInstanceTransform(id, transform))
      if prev.projection4D != spec.projection4D then
        val p = spec.projection4D.getOrElse(Projection4DSpec.default)
        state.slotsPerSpec(i).foreach { slot =>
          val _ = renderer.updateMesh4DProjection(
            slot, eyeW = p.eyeW, screenW = p.screenW, rotXW = p.rotXW, rotYW = p.rotYW, rotZW = p.rotZW
          )
        }
      for
        level <- spec.level if !prev.level.contains(level)
        caps <- state.capsInstancePerSpec(i)
      do
        val _ = renderer.setInstanceCoverage(caps, TriangleMeshSceneBuilder.holeCapsCoverage(level))
    }
    state.copy(specs = next)
