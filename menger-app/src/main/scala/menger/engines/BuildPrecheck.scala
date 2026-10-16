package menger.engines

import scala.util.Failure
import scala.util.Success
import scala.util.Try

import menger.ObjectSpec
import menger.common.Const
import menger.common.ProfilingConfig

/** Every reason the renderer would refuse to build `specs`, found without touching it: the
  * renderer's own grouping (`SceneGroups`, `RenderModeSelector`) and each scene builder's
  * `validate` -- pure CPU checks. Shared by the validator (a scene finding instead of a dead
  * window, usability review 2026-09, F19) and the windows' live reload, which checks before it
  * tears the current scene down (review 2026-10-14, menger#101). The instance limit is the
  * ceiling the engines auto-adjust up to. */
object BuildPrecheck:

  def problems(specs: List[ObjectSpec])(using ProfilingConfig): List[String] =
    if specs.isEmpty then Nil
    else RenderModeSelector.classify(specs) match
      case SceneType.Unsupported(_) =>
        val types = specs.map(_.objectType).distinct.mkString(", ")
        List(s"analytical primitives can be mixed with one mesh type at a time, got: $types")
      case sceneType =>
        val groups = sceneType match
          case _ if SceneGroups.hasMixedEdge4D(specs) => SceneGroups.buildOrder(specs)
          case SceneType.SimpleMixed(_, _) => SceneGroups.buildOrder(specs)
          case _ => List(specs)
        groups.flatMap(groupProblem)

  private def groupProblem(group: List[ObjectSpec])(using ProfilingConfig): Option[String] =
    val types = group.map(_.objectType).distinct.mkString(", ")
    GeometryRegistry.builderFor(group) match
      case None => Some(s"the renderer has no scene builder for: $types")
      case Some(builder) =>
        Try(builder.validate(group, Const.maxInstancesLimit)) match
          case Success(Right(()))    => None
          case Success(Left(reason)) => Some(s"$types: $reason")
          case Failure(e)            => Some(s"$types: ${e.getMessage}")
