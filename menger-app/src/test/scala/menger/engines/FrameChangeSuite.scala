package menger.engines

import menger.common.Axis
import menger.common.Const
import menger.common.PlaneSpec
import menger.config.PlaneConfig
import menger.spec.ObjectRotation
import menger.spec.ObjectSpec
import menger.spec.Radians
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// menger#86: a frame whose objects did not change was rebuilt from scratch anyway.
class FrameChangeSuite extends AnyFlatSpec with Matchers:

  private val sponge = ObjectSpec("sponge-volume", level = Some(2f))
  private def geometry(
    specs: List[ObjectSpec] = List(sponge),
    planes: List[PlaneConfig] = Nil,
    envMap: Option[String] = None,
    animatedTextures: Boolean = false
  ) = FrameChange.Geometry(specs, planes, envMap, animatedTextures)

  "FrameChange.classify" should "rebuild the first frame" in:
    FrameChange.classify(None, geometry()) shouldBe FrameChange.Rebuild

  it should "leave a frame with the same objects, planes and environment map alone" in:
    FrameChange.classify(Some(geometry()), geometry()) shouldBe FrameChange.Unchanged

  it should "rebuild when an object changes" in:
    val turned = sponge.copy(rotation = ObjectRotation(y = Radians(0.5f)))
    FrameChange.classify(Some(geometry()), geometry(List(turned))) shouldBe FrameChange.Rebuild
    val deeper = sponge.copy(level = Some(3f))
    FrameChange.classify(Some(geometry()), geometry(List(deeper))) shouldBe FrameChange.Rebuild

  it should "rebuild when an object is added or removed" in:
    FrameChange.classify(Some(geometry()), geometry(List(sponge, sponge))) shouldBe
      FrameChange.Rebuild
    FrameChange.classify(Some(geometry()), geometry(Nil)) shouldBe FrameChange.Rebuild

  it should "rebuild when a plane changes" in:
    val floor = PlaneConfig(PlaneSpec(Axis.Y, false, Const.defaultFloorPlaneY), colorSpec = None)
    FrameChange.classify(Some(geometry()), geometry(planes = List(floor))) shouldBe
      FrameChange.Rebuild

  it should "rebuild when the environment map changes" in:
    FrameChange.classify(Some(geometry()), geometry(envMap = Some("sky.hdr"))) shouldBe
      FrameChange.Rebuild

  it should "always rebuild a frame with a video texture" in:
    val video = geometry(animatedTextures = true)
    FrameChange.classify(Some(video), video) shouldBe FrameChange.Rebuild
