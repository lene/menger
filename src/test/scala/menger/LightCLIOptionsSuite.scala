package menger

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.rogach.scallop.exceptions.ScallopException

class LightCLIOptionsSuite extends AnyFlatSpec with Matchers:
  class SafeMengerCLIOptions(args: Seq[String]) extends menger.MengerCLIOptions(args):
    @SuppressWarnings(Array("org.wartremover.warts.Throw"))
    override def onError(e: Throwable): Unit = throw e

  "--light" should "default to None when not provided" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere"))
    opts.light.toOption shouldBe None

  it should "parse a single directional light with position only" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "directional:1.0,2.0,3.0"))
    opts.light.toOption shouldBe defined
    val lights = opts.light()
    lights should have size 1
    lights.head.lightType shouldBe LightType.DIRECTIONAL
    lights.head.position shouldEqual Vector3(1.0f, 2.0f, 3.0f)
    lights.head.intensity shouldEqual 1.0f
    lights.head.color shouldEqual Color.WHITE

  it should "parse a single point light with position only" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "point:0.0,-5.0,10.0"))
    opts.light.toOption shouldBe defined
    val lights = opts.light()
    lights should have size 1
    lights.head.lightType shouldBe LightType.POINT
    lights.head.position shouldEqual Vector3(0.0f, -5.0f, 10.0f)
    lights.head.intensity shouldEqual 1.0f
    lights.head.color shouldEqual Color.WHITE

  it should "parse a directional light with intensity" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "directional:-1.0,1.0,-1.0:2.5"))
    val lights = opts.light()
    lights should have size 1
    lights.head.lightType shouldBe LightType.DIRECTIONAL
    lights.head.position shouldEqual Vector3(-1.0f, 1.0f, -1.0f)
    lights.head.intensity shouldEqual 2.5f
    lights.head.color shouldEqual Color.WHITE

  it should "parse a point light with intensity" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "point:0.0,5.0,0.0:0.5"))
    val lights = opts.light()
    lights should have size 1
    lights.head.lightType shouldBe LightType.POINT
    lights.head.position shouldEqual Vector3(0.0f, 5.0f, 0.0f)
    lights.head.intensity shouldEqual 0.5f
    lights.head.color shouldEqual Color.WHITE

  it should "parse a light with hex color (no intensity)" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "directional:1.0,0.0,0.0::ff0000"))
    val lights = opts.light()
    lights should have size 1
    lights.head.lightType shouldBe LightType.DIRECTIONAL
    lights.head.intensity shouldEqual 1.0f
    lights.head.color shouldEqual Color.valueOf("ff0000")

  it should "parse a light with intensity and hex color" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "point:0.0,5.0,0.0:2.0:00ff00"))
    val lights = opts.light()
    lights should have size 1
    lights.head.lightType shouldBe LightType.POINT
    lights.head.intensity shouldEqual 2.0f
    lights.head.color shouldEqual Color.valueOf("00ff00")

  it should "parse a light with intensity and RGB color" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "directional:0.0,1.0,0.0:1.5:255,0,0"))
    val lights = opts.light()
    lights should have size 1
    lights.head.intensity shouldEqual 1.5f
    lights.head.color.r shouldEqual 1.0f +- 0.01f
    lights.head.color.g shouldEqual 0.0f +- 0.01f
    lights.head.color.b shouldEqual 0.0f +- 0.01f

  it should "parse multiple lights (repeatable flag)" in:
    val opts = SafeMengerCLIOptions(Seq(
      "--optix", "--sponge-type", "sphere",
      "--light", "directional:1.0,0.0,0.0",
      "--light", "point:0.0,5.0,0.0:2.0:ffffff"
    ))
    val lights = opts.light()
    lights should have size 2
    lights(0).lightType shouldBe LightType.DIRECTIONAL
    lights(0).position shouldEqual Vector3(1.0f, 0.0f, 0.0f)
    lights(1).lightType shouldBe LightType.POINT
    lights(1).position shouldEqual Vector3(0.0f, 5.0f, 0.0f)
    lights(1).intensity shouldEqual 2.0f

  it should "parse up to 8 lights" in:
    val lightArgs = Seq("--optix", "--sponge-type", "sphere") ++
      (1 to 8).flatMap(i => Seq("--light", s"directional:$i.0,0.0,0.0"))
    val opts = SafeMengerCLIOptions(lightArgs)
    val lights = opts.light()
    lights should have size 8

  it should "reject more than 8 lights" in:
    val lightArgs = Seq("--optix", "--sponge-type", "sphere") ++
      (1 to 9).flatMap(i => Seq("--light", s"directional:$i.0,0.0,0.0"))
    an[ScallopException] should be thrownBy SafeMengerCLIOptions(lightArgs)

  it should "reject --light without --optix flag" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--light", "directional:1.0,0.0,0.0"))

  it should "reject invalid light type" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "invalid:1.0,0.0,0.0"))

  it should "reject malformed light specification (missing position)" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "directional:"))

  it should "reject malformed light specification (incomplete position)" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "directional:1.0,2.0"))

  it should "reject malformed light specification (non-numeric position)" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "directional:x,y,z"))

  it should "reject malformed light specification (non-numeric intensity)" in:
    an[ScallopException] should be thrownBy
      SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "directional:1.0,0.0,0.0:bright"))

  it should "accept negative position values" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "directional:-1.0,-2.0,-3.0"))
    val lights = opts.light()
    lights.head.position shouldEqual Vector3(-1.0f, -2.0f, -3.0f)

  it should "accept negative intensity values" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "directional:1.0,0.0,0.0:-0.5"))
    val lights = opts.light()
    lights.head.intensity shouldEqual -0.5f

  it should "accept zero intensity" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "point:0.0,5.0,0.0:0.0"))
    val lights = opts.light()
    lights.head.intensity shouldEqual 0.0f

  it should "parse case-insensitive light types" in:
    val opts1 = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "DIRECTIONAL:1.0,0.0,0.0"))
    opts1.light().head.lightType shouldBe LightType.DIRECTIONAL

    val opts2 = SafeMengerCLIOptions(Seq("--optix", "--sponge-type", "sphere", "--light", "Point:0.0,5.0,0.0"))
    opts2.light().head.lightType shouldBe LightType.POINT
