package menger.engines.scene

import io.github.lene.optix.OptiXRenderer
import menger.common.Material
import menger.common.Vector
import menger.engines.scene.AxialPrimitiveSceneBuilder.ExplicitAxis
import menger.spec.ObjectSpec

/** Scene builder for capped cylinders: `p0` and `p1` are the centres of the end caps. optix-jni
  * shades a cylinder opaque; `SceneValidator` reports a transparent material on one. */
class CylinderSceneBuilder(textureDir: String = ".")
    extends AxialPrimitiveSceneBuilder("cylinder", textureDir):

  override private[scene] def explicitAxisOf(spec: ObjectSpec): ExplicitAxis =
    ExplicitAxis(top = spec.p1, bottom = spec.p0, radius = spec.cylinderRadius)

  override protected def addInstance(
    renderer: OptiXRenderer, top: Vector[3], bottom: Vector[3], radius: Float, material: Material
  ): InstanceId =
    renderer.addCylinderInstance(bottom, top, radius, material)
