package menger.engines.scene

import io.github.lene.optix.OptiXRenderer
import menger.common.Material
import menger.common.Vector
import menger.engines.scene.AxialPrimitiveSceneBuilder.ExplicitAxis
import menger.spec.ObjectSpec

/** Scene builder for cones: apex on top, base centre at the bottom, radius of the base. */
class ConeSceneBuilder(textureDir: String = ".")
    extends AxialPrimitiveSceneBuilder("cone", textureDir):

  override private[scene] def explicitAxisOf(spec: ObjectSpec): ExplicitAxis =
    ExplicitAxis(top = spec.apex, bottom = spec.base, radius = spec.radius)

  override protected def addInstance(
    renderer: OptiXRenderer, top: Vector[3], bottom: Vector[3], radius: Float, material: Material
  ): InstanceId =
    renderer.addConeInstance(top, bottom, radius, material)
