package menger.engines.scene

import menger.objects.higher_d.LiftedProjection
import menger.objects.higher_d.Plane
import menger.objects.higher_d.Projection
import menger.spec.ProjectionSpec

/** A scene's [[ProjectionSpec]] as the `menger.objects` types that apply it: `menger.spec` does
  * not see `menger.objects`, so the plane letters become [[Plane]] indices here (menger#162). */
object ProjectionConversion:

  def liftedProjection(spec: ProjectionSpec): LiftedProjection =
    val turns = spec.planes.map { rotation =>
      (Plane(rotation.plane.firstAxis, rotation.plane.secondAxis, spec.dimension),
        rotation.angle.value)
    }
    LiftedProjection(
      spec.dimension, turns, spec.stagePairs.map(stage => Projection(stage.eye, stage.screen))
    )
