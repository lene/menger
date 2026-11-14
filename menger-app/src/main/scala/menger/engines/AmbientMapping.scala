package menger.engines

import io.github.lene.optix.AmbientModel
import menger.dsl.Ambient

/** The DSL's ambient choice as optix-jni's model. It lives here, not on `Ambient`, because a
  * scene is compiled against the DSL surface alone, which must not reach into the renderer. */
object AmbientMapping:

  def toModel(ambient: Ambient): AmbientModel = ambient match
    case Ambient.Constant                     => AmbientModel.Constant
    case Ambient.Occlusion(samples, radius)   => AmbientModel.Occlusion(samples, radius)
