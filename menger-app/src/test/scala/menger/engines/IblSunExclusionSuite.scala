package menger.engines

import scala.language.implicitConversions

import io.github.lene.optix.OptiXRenderer
import io.github.lene.optix.TextureIndex
import menger.common.Vector
import menger.config.IblSunExclusion
import menger.dsl.Camera
import menger.dsl.Caustics
import menger.dsl.Color
import menger.dsl.Directional
import menger.dsl.EnvMapSunMeasurement
import menger.dsl.IBL
import menger.dsl.Light
import menger.dsl.Scene
import menger.dsl.Sphere
import menger.dsl.Sun
import menger.dsl.Vec3
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class IblSunExclusionSuite extends AnyFlatSpec with Matchers with MockFactory:

  private val fallbackCaustics = Caustics.Disabled.toCausticsConfig
  private val sky = "assets/hdri/sky.hdr"
  private val towardSun = Vector[3](0f, 0.6f, 0.8f)
  private val halfAngle = 2f
  private val measurement =
    EnvMapSunMeasurement(Vec3(0f, 0.6f, 0.8f), halfAngle, 4.3f, Color(1f, 0.9f, 0.7f))
  private val measureAnyMap: String => Option[EnvMapSunMeasurement] = _ => Some(measurement)
  private val exclusion = IblSunExclusion(towardSun, halfAngle)

  private def sceneWith(
    lights: List[Light], envMap: Option[String] = Some(sky), ibl: Option[IBL] = Some(IBL())
  ): Scene =
    Scene(Camera.Default, List(Sphere(menger.dsl.Material.Glass)), lights, envMap = envMap, ibl = ibl)

  private def sunOf(map: String): Sun = Sun(Vec3(0f, -0.6f, -0.8f), measuredFromEnvMap = Some(map))

  "SceneConverter.convert" should "exclude the sun of the env map when a Sun measured from it lights the scene" in:
    val configs = SceneConverter.convert(sceneWith(List(sunOf(sky))), fallbackCaustics, measureAnyMap)

    configs.iblSunExclusion shouldBe Some(exclusion)

  it should "ask for the measurement of the scene's map, not of another one" in:
    val asked = new java.util.concurrent.CopyOnWriteArrayList[String]()
    SceneConverter.convert(
      sceneWith(List(sunOf(sky))), fallbackCaustics, path => { asked.add(path); None })

    asked.toArray.toList shouldBe List(sky)

  it should "not exclude anything without image-based lighting" in:
    val scene = sceneWith(List(sunOf(sky)), ibl = None)

    SceneConverter.convert(scene, fallbackCaustics, measureAnyMap).iblSunExclusion shouldBe None

  it should "not exclude the sun of another map than the scene's" in:
    val scene = sceneWith(List(sunOf("assets/hdri/other.hdr")))

    SceneConverter.convert(scene, fallbackCaustics, measureAnyMap).iblSunExclusion shouldBe None

  it should "not exclude anything for a Sun that was not measured from a map" in:
    val scene = sceneWith(List(Sun(Vec3(0f, -1f, 0f))))

    SceneConverter.convert(scene, fallbackCaustics, measureAnyMap).iblSunExclusion shouldBe None

  it should "not exclude anything for a scene without a Sun" in:
    val scene = sceneWith(List(Directional(Vec3(0f, -1f, 0f))))

    SceneConverter.convert(scene, fallbackCaustics, measureAnyMap).iblSunExclusion shouldBe None

  "IblSetup.applySunExclusion" should "hand the cone to the renderer" in:
    val renderer = mock[OptiXRenderer]
    (renderer.setIBLSunExclusion _).expects(towardSun, halfAngle).once()

    IblSetup.applySunExclusion(renderer, Some(exclusion))

  it should "end the exclusion when the scene has none" in:
    val renderer = mock[OptiXRenderer]
    (renderer.clearIBLSunExclusion _).expects().once()

    IblSetup.applySunExclusion(renderer, None)

  // menger#139: the preview and the CLI animation skipped the scene's image-based lighting.
  "IblSetup.applyEnvironment" should "set the scene's env map and image-based lighting" in:
    val configs = SceneConverter.convert(
      sceneWith(List(Directional(Vec3(0f, -1f, 0f)))), fallbackCaustics, _ => None)
    val renderer = mock[OptiXRenderer]
    val skyIndex = TextureIndex(7)
    (renderer.uploadTextureFromFile _).expects(where((path: String) => path.endsWith(sky)))
      .returning(skyIndex).once()
    (renderer.setEnvironmentMap _).expects(skyIndex).once()
    (renderer.setIBL _).expects(true, configs.iblStrength, configs.iblSamples).once()
    (renderer.clearIBLSunExclusion _).expects().once()

    IblSetup.applyEnvironment(renderer, ".", configs)

  it should "leave image-based lighting alone for a scene without it" in:
    val configs = SceneConverter.convert(
      sceneWith(List(Directional(Vec3(0f, -1f, 0f))), envMap = None, ibl = None), fallbackCaustics)
    val renderer = mock[OptiXRenderer]
    (renderer.setIBL _).expects(*, *, *).never()
    (renderer.setEnvironmentMap _).expects(*).never()

    IblSetup.applyEnvironment(renderer, ".", configs)

  // menger#139: a live reload rebuilt the scene without its sky.
  "IblSetup.uploadEnvMaps" should "set the env map again for every rebuild" in:
    val renderer = mock[OptiXRenderer]
    val skyIndex = TextureIndex(7)
    (renderer.uploadTextureFromFile _).expects(where((path: String) => path.endsWith(sky)))
      .returning(skyIndex).twice()
    (renderer.setEnvironmentMap _).expects(skyIndex).twice()

    IblSetup.uploadEnvMaps(renderer, ".", Some(sky), None, 0f)
    IblSetup.uploadEnvMaps(renderer, ".", Some(sky), None, 0f)

  it should "touch nothing for a scene without a sky" in:
    val renderer = mock[OptiXRenderer]
    (renderer.uploadTextureFromFile _).expects(*).never()
    (renderer.setEnvironmentMap _).expects(*).never()

    IblSetup.uploadEnvMaps(renderer, ".", None, None, 0f)

  // A plain rebuild keeps the uploaded texture: not an HDR read per step of a 4D rotation.
  "IblSetup.applySky" should "upload the sky once and not again for a plain rebuild" in:
    val renderer = mock[OptiXRenderer]
    (() => renderer.maxInstances).expects().returning(64).anyNumberOfTimes()
    (renderer.uploadTextureFromFile _).expects(*).returning(TextureIndex(7)).once()
    (renderer.setEnvironmentMap _).expects(TextureIndex(7)).once()

    val first = IblSetup.applySky(renderer, ".", Some(sky), None, 0f, None)
    val second = IblSetup.applySky(renderer, ".", Some(sky), None, 0f, first)
    val third = IblSetup.applySky(renderer, ".", Some(sky), None, 0f, second)

    third shouldBe first

  it should "upload again after the renderer was reinitialized with a larger capacity" in:
    val renderer = mock[OptiXRenderer]
    val capacities = Iterator(64, 64, 2048)
    (() => renderer.maxInstances).expects().onCall(() => capacities.next()).repeat(3)
    (renderer.uploadTextureFromFile _).expects(*).returning(TextureIndex(7)).twice()
    (renderer.setEnvironmentMap _).expects(TextureIndex(7)).twice()

    val first = IblSetup.applySky(renderer, ".", Some(sky), None, 0f, None)
    val rebuilt = IblSetup.applySky(renderer, ".", Some(sky), None, 0f, first)
    val reinitialized = IblSetup.applySky(renderer, ".", Some(sky), None, 0f, rebuilt)

    reinitialized.map(_.rendererCapacity) shouldBe Some(2048)

  it should "upload the new map when a reload changed the scene's env map" in:
    val renderer = mock[OptiXRenderer]
    (() => renderer.maxInstances).expects().returning(64).anyNumberOfTimes()
    (renderer.uploadTextureFromFile _).expects(where((path: String) => path.endsWith(sky)))
      .returning(TextureIndex(7)).once()
    (renderer.uploadTextureFromFile _).expects(where((path: String) => path.endsWith("other.hdr")))
      .returning(TextureIndex(8)).once()
    (renderer.setEnvironmentMap _).expects(TextureIndex(7)).once()
    (renderer.setEnvironmentMap _).expects(TextureIndex(8)).once()

    val first = IblSetup.applySky(renderer, ".", Some(sky), None, 0f, None)
    val reloaded = IblSetup.applySky(renderer, ".", Some("assets/hdri/other.hdr"), None, 0f, first)

    reloaded.flatMap(_.envMap) shouldBe Some("assets/hdri/other.hdr")
