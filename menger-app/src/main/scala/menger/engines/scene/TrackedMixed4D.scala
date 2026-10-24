package menger.engines.scene

import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.common.ObjectType
import menger.common.ProfilingConfig
import menger.engines.SceneGroups

/** A 4D scene of edge-rendered and plain projected meshes, each part tracked as its own kind
  * (TrackedEdges4D, TrackedMesh4D), so that a 4D rotation moves both in place. Without it, such
  * a scene had no in-place path and every animation frame was a full rebuild (usability review
  * 2026-10, session 6, menger#129: a TesseractSponge with edge tubes and a 24-cell without,
  * 1.12 s per frame, 1.0 s of it the rebuild). */
object TrackedMixed4D:

  final case class State(edges: TrackedEdges4D.State, meshes: TrackedMesh4D.State)

  // ponytail: projected 4D meshes only; a mixed scene with anything else (a sphere, a 3D mesh)
  // still rebuilds every frame. Track the other SceneGroups.buildOrder groups when one is slow.
  def isMixedScene(specs: List[ObjectSpec]): Boolean =
    SceneGroups.hasMixedEdge4D(specs) && specs.forall { s =>
      val t = ObjectType.normalize(s.objectType)
      ObjectType.isTriangleMesh(t) && ObjectType.isProjected4D(t)
    }

  /** Builds the edged part first, as `SceneGroups.buildOrder` does (its builder may reinitialize
    * the renderer); both parts get the scene's `maxInstances`, as in `buildMixedSceneObjects`.
    * `None` in the result when either part cannot be moved in place. */
  def build(
    specs: List[ObjectSpec],
    renderer: OptiXRenderer,
    textureDir: String,
    maxInstances: Int
  )(using ProfilingConfig): Try[Option[State]] =
    val (edged, plain) = specs.partition(_.hasEdgeRendering)
    for
      edges <- TrackedEdges4D.build(edged, renderer, textureDir, _ => maxInstances)
      meshes <- TrackedMesh4D.build(plain, renderer, textureDir, _ => maxInstances)
    yield edges.zip(meshes).map(State.apply)

  /** `next` applied in place when both parts can take it; `None` (nothing changed on the
    * renderer) when the scene must be rebuilt. The mesh part is checked first, since the edge
    * part checks and moves in one step. */
  def update(state: State, next: List[ObjectSpec], renderer: OptiXRenderer): Option[State] =
    val (edged, plain) = next.partition(_.hasEdgeRendering)
    if !TrackedMesh4D.canUpdateInPlace(state.meshes.specs, plain) then None
    else TrackedEdges4D.update(state.edges, edged, renderer).map { edges =>
      State(edges, TrackedMesh4D.updateInPlace(state.meshes, plain, renderer))
    }
