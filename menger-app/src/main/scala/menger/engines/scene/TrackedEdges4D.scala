package menger.engines.scene

import scala.collection.mutable
import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.common.ProfilingConfig
import menger.spec.ObjectSpec
import menger.spec.ObjectType

/** An edge-rendered 4D scene built with its cylinders and faces recorded, so that a change of
  * 4D projection moves them in place (TesseractEdgeSceneBuilder.updateProjection) instead of
  * rebuilding every cylinder (usability review 2026-10, session 5, F112, menger#129: a 4D
  * rotation animation of a sponge with edge tubes rebuilt 27,648 cylinders per frame, 3.2 s
  * each). The interactive window had this since session 2 (F22); the animated window and the
  * frame/video export now share it. */
object TrackedEdges4D:

  final case class State(specs: List[ObjectSpec], tracks: IndexedSeq[TesseractEdgeSceneBuilder.EdgeTrack])

  /** The scenes TesseractEdgeSceneBuilder takes (GeometryRegistry.builderFor): projected 4D
    * triangle meshes, every one of them edge-rendered. */
  def isEdgeScene(specs: List[ObjectSpec]): Boolean =
    specs.nonEmpty && specs.forall { s =>
      val t = ObjectType.normalize(s.objectType)
      ObjectType.isTriangleMesh(t) && ObjectType.isProjected4D(t) && s.hasEdgeRendering
    }

  /** Builds `specs` with a recording TesseractEdgeSceneBuilder; `None` in the result when some
    * spec recorded no track, so the scene cannot be moved in place. */
  def build(
    specs: List[ObjectSpec],
    renderer: OptiXRenderer,
    textureDir: String,
    maxInstances: SceneBuilder => Int
  )(using ProfilingConfig): Try[Option[State]] =
    val tracks = mutable.Map.empty[Int, TesseractEdgeSceneBuilder.EdgeTrack]
    val builder = TesseractEdgeSceneBuilder(textureDir, (spec, track) => tracks(spec) = track)
    builder.validateAndBuild(specs, renderer, maxInstances(builder)).map { _ =>
      Option.when(tracks.size == specs.size)(State(specs, specs.indices.map(tracks)))
    }

  /** `next` applied in place when only 4D projections changed and the eye-w clip set is the
    * same; `None` (nothing changed on the renderer) when the scene must be rebuilt. */
  def update(state: State, next: List[ObjectSpec], renderer: OptiXRenderer): Option[State] =
    Option.when(TesseractEdgeSceneBuilder.updateProjection(state.specs, next, state.tracks, renderer))(
      state.copy(specs = next)
    )
