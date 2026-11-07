package menger.dsl

import menger.common.ImageSize
import menger.common.ProfilingConfig
import menger.common.Vector
import menger.common.{Color => CommonColor}
import menger.common.{Material => CommonMaterial}
import menger.engines.scene.GPURequired
import menger.geometry.MengerRenderer
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** `Sun(intensity = k)` lights the origin like `Directional(intensity = k)` whatever its
  * distance, so only the penumbra changes with it (menger#158). GPU-required: cancels without
  * an OptiX device. */
class SunRenderSuite extends AnyFlatSpec with Matchers:

  private given ProfilingConfig = ProfilingConfig.disabled

  private val Size = ImageSize(160, 120)
  private val FloorY = -0.25f
  private val FloorGrey = 0.8f
  private val CentreFraction = 0.25
  private val SameBrightness = 0.02
  private val MinBrightness = 20.0
  private val TestIntensity = 0.5f // leaves headroom below clipping
  private val SunDirection = Vec3(0.3f, -1f, 0.2f)
  private val LowSunDirection = Vec3(0.8f, -0.5f, 0f)
  private val SceneRadius = 4f

  private def floorBrightness(light: Light, at: (Float, Float) = (0f, 0f)): Double =
    val renderer =
      try
        val r = MengerRenderer()
        r.initialize()
        r
      catch case _: Throwable => cancel("OptiX native library not available")
    try
      renderer.setAntialiasing(false, 1, 0.1f)
      renderer.setCamera(
        Vector[3](at._1, 6f, at._2 + 0.5f), Vector[3](at._1, FloorY, at._2), Vector[3](0f, 1f, 0f),
        45f
      )
      val grey = CommonColor(FloorGrey, FloorGrey, FloorGrey)
      renderer.addPlaneInstance(Vector[3](0f, 1f, 0f), FloorY, CommonMaterial.matte(grey))
      renderer.setLights(Array(light.toCommonLight))
      renderer.setShadows(true)
      centreMean(renderer.render(Size))
    finally renderer.dispose()

  private def centreMean(rgba: Array[Byte]): Double =
    val (halfW, halfH) = ((Size.width * CentreFraction).toInt, (Size.height * CentreFraction).toInt)
    val samples = for
      y <- (Size.height / 2 - halfH) until (Size.height / 2 + halfH)
      x <- (Size.width / 2 - halfW) until (Size.width / 2 + halfW)
      c <- 0 until 3
    yield (rgba((y * Size.width + x) * 4 + c) & 0xFF).toDouble
    samples.sum / samples.length

  "A Sun" should "light the floor equally bright at distance 20 and 80" taggedAs GPURequired in:
    val near = floorBrightness(Sun(SunDirection, distance = 20f, intensity = TestIntensity))
    val far = floorBrightness(Sun(SunDirection, distance = 80f, intensity = TestIntensity))
    info(f"lit floor: $near%.2f at 20, $far%.2f at 80")
    near should be > MinBrightness
    math.abs(near - far) / near should be < SameBrightness

  it should "light the floor like a Directional of the same intensity" taggedAs GPURequired in:
    val sun = floorBrightness(Sun(SunDirection, intensity = TestIntensity))
    val directional = floorBrightness(Directional(SunDirection, intensity = TestIntensity))
    info(f"lit floor: $sun%.2f for the Sun, $directional%.2f for the Directional")
    math.abs(sun - directional) / directional should be < SameBrightness

  it should "light a point 4 units toward the light like the origin by default" in:
    val sun = Sun(LowSunDirection, intensity = TestIntensity)
    val origin = floorBrightness(sun)
    val towardLight = floorBrightness(sun, at = (-SceneRadius, 0f))
    info(f"lit floor: $origin%.2f at the origin, $towardLight%.2f 4 units toward the light")
    origin should be > MinBrightness
    math.abs(towardLight - origin) / origin should be < SameBrightness
