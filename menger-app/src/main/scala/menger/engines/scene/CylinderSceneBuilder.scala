package menger.engines.scene

import io.github.lene.optix.OptiXRenderer
import menger.common.Material
import menger.common.Vector
import menger.engines.scene.AxialPrimitiveSceneBuilder.ExplicitAxis
import menger.spec.ObjectSpec

/** Scene builder for capped cylinders: the axis runs from `p0` to `p1`, the centres of the end
  * caps. optix-jni shades a cylinder opaque; `SceneValidator` reports a material it cannot draw
  * as written. */
class CylinderSceneBuilder(textureDir: String = ".")
    extends AxialPrimitiveSceneBuilder("cylinder", textureDir):

  override private[scene] def explicitAxisOf(spec: ObjectSpec): ExplicitAxis =
    ExplicitAxis(start = spec.p0, end = spec.p1, radius = spec.cylinderRadius)

  override protected def addInstance(
    renderer: OptiXRenderer, start: Vector[3], end: Vector[3], radius: Float, material: Material
  ): InstanceId =
    renderer.addCylinderInstance(start, end, radius, material)
