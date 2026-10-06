package menger.engines.scene

import menger.ObjectRotation
import menger.ObjectSpec
import menger.common.TransformUtil
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** optix-jni#61 (F63, F67): the object frame maps the object's own box onto [0,1]^3. */
class ObjectFrameSpec extends AnyFlatSpec with Matchers:

  private def apply(frame: Array[Float], p: (Float, Float, Float)): Seq[Float] =
    (0 until 3).map(i => frame(4 * i) * p._1 + frame(4 * i + 1) * p._2 + frame(4 * i + 2) * p._3 +
      frame(4 * i + 3))

  private def near(a: Seq[Float], b: Seq[Float]) =
    a.zip(b).foreach((x, y) => x shouldBe y +- 1e-5f)

  "SceneBuilder.objectFrame" should "map a cube's corners to 0 and 1" in:
    val frame = SceneBuilder.objectFrame(ObjectSpec("cube", x = 1f, y = 1f, z = 1f, size = 2f)).get
    near(apply(frame, (0f, 0f, 0f)), Seq(0f, 0f, 0f))
    near(apply(frame, (2f, 2f, 2f)), Seq(1f, 1f, 1f))
    near(apply(frame, (1f, 1f, 1f)), Seq(0.5f, 0.5f, 0.5f))

  it should "use a sphere's size as its radius" in:
    val frame = SceneBuilder.objectFrame(ObjectSpec("sphere", size = 1.5f)).get
    near(apply(frame, (1.5f, -1.5f, 0f)), Seq(1f, 0f, 0.5f))

  it should "follow the object's rotation" in:
    // The corner the instance transform puts at world w must come back as local (1, 0, 1).
    val spec = ObjectSpec("cube", x = 3f, size = 2f, rotation = ObjectRotation(0.3f, 1.1f, -0.4f))
    val m = TransformUtil.createEulerRotationScaleTranslation(
      spec.rotX, spec.rotY, spec.rotZ, 1f, spec.x, spec.y, spec.z
    )
    val corner = (1f, -1f, 1f)
    val world = (0 until 3).map(i =>
      m(4 * i) * corner._1 + m(4 * i + 1) * corner._2 + m(4 * i + 2) * corner._3 + m(4 * i + 3))
    near(apply(SceneBuilder.objectFrame(spec).get, (world(0), world(1), world(2))), Seq(1f, 0f, 1f))

  it should "give a plane no frame" in:
    SceneBuilder.objectFrame(ObjectSpec("plane")) shouldBe None
