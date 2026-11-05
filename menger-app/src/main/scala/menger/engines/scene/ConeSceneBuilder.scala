package menger.engines.scene

import io.github.lene.optix.OptiXRenderer
import menger.common.Material
import menger.common.Vector
import menger.engines.scene.AxialPrimitiveSceneBuilder.ExplicitAxis
import menger.spec.ObjectSpec

/** Scene builder for cones: the axis runs from the base centre to the apex; the radius is the
  * base's. */
class ConeSceneBuilder(textureDir: String = ".")
    extends AxialPrimitiveSceneBuilder("cone", textureDir):

  override private[scene] def explicitAxisOf(spec: ObjectSpec): ExplicitAxis =
    ExplicitAxis(start = spec.base, end = spec.apex, radius = spec.radius)

  override protected def addInstance(
    renderer: OptiXRenderer, start: Vector[3], end: Vector[3], radius: Float, material: Material
  ): InstanceId =
    renderer.addConeInstance(end, start, radius, material)
