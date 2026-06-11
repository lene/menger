package menger.dsl

import scala.language.implicitConversions

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.engines.SceneConverter

/** Integration tests demonstrating end-to-end scene creation and rendering preparation. */
class SceneIntegrationSuite extends AnyFlatSpec with Matchers:

  private val fallback = Caustics.Disabled.toCausticsConfig
  private def convert(scene: Scene) = SceneConverter.convert(scene, fallback)


  "DSL Scene" should "create a complete glass sphere scene ready for rendering" in:
    // Define scene using DSL
    val scene = Scene(
      camera = Camera((0f, 0f, 3f), (0f, 0f, 0f)),
      objects = List(
        Sphere(Material.Glass)
      ),
      lights = List(
        Directional((1f, -1f, -1f))
      )
    )

    // Convert to rendering configuration
    val converted = convert(scene)
    val sceneConfig = converted.scene
    val cameraConfig = converted.camera

    // Verify scene is properly configured
    sceneConfig.isMultiObject shouldBe true
    sceneConfig.isEmpty shouldBe false

    val specs = sceneConfig.objectSpecs.get
    specs should have length 1
    specs(0).objectType shouldBe "sphere"
    specs(0).material shouldBe defined

    // Verify camera is properly configured
    cameraConfig.position.z shouldBe 3f
    cameraConfig.lookAt.z shouldBe 0f

  it should "create a multi-object scene with various materials" in:
    val scene = Scene(
      camera = Camera((0f, 2f, 5f), (0f, 0f, 0f)),
      objects = List(
        Sphere((-2f, 0f, 0f), Material.Glass, 1.0f),
        Cube((0f, 0f, 0f), Material.Gold, 1.0f),
        Sphere((2f, 0f, 0f), Material.Chrome, 1.0f)
      ),
      lights = List(
        Directional((1f, -1f, -1f), 1.5f),
        Point((0f, 5f, 0f), 2.0f)
      ),
      caustics = Caustics.Default
    )

    val sceneConfig = convert(scene).scene
    val specs = sceneConfig.objectSpecs.get

    specs should have length 3
    specs(0).objectType shouldBe "sphere"
    specs(1).objectType shouldBe "cube"
    specs(2).objectType shouldBe "sphere"

    scene.lights should have length 2
    scene.caustics shouldBe defined

  it should "create a Menger sponge scene" in:
    val scene = Scene(
      camera = Camera((0f, 0f, 4f), (0f, 0f, 0f)),
      objects = List(
        Sponge(VolumeFilling, level = 2f, Material.Chrome, size = 2.0f)
      ),
      lights = List(
        Directional((1f, -1f, -1f))
      )
    )

    val sceneConfig = convert(scene).scene
    val specs = sceneConfig.objectSpecs.get

    specs should have length 1
    specs(0).objectType shouldBe "sponge-volume"
    specs(0).level shouldBe Some(2f)
    specs(0).size shouldBe 2.0f

  it should "create a scene with colored objects without materials" in:
    val scene = Scene(
      camera = Camera.Default,
      objects = List(
        Sphere(pos = Vec3(-1f, 0f, 0f), color = Some(Color.Red)),
        Sphere(pos = Vec3(1f, 0f, 0f), color = Some(Color.Blue))
      ),
      lights = List(
        Point((0f, 3f, 0f))
      )
    )

    val sceneConfig = convert(scene).scene
    val specs = sceneConfig.objectSpecs.get

    specs should have length 2
    specs(0).color shouldBe defined
    specs(1).color shouldBe defined

  it should "create a scene with textured objects" in:
    val scene = Scene(
      camera = Camera.Default,
      objects = List(
        Cube(texture = Some("brick.png")),
        Sphere(Vec3(2f, 0f, 0f), texture = Some("metal.png"))
      ),
      lights = List(
        Directional((1f, -1f, 0f))
      )
    )

    val sceneConfig = convert(scene).scene
    val specs = sceneConfig.objectSpecs.get

    specs(0).texture shouldBe Some("brick.png")
    specs(1).texture shouldBe Some("metal.png")

  it should "create a scene with video-textured objects" in:
    val videoTexture = VideoTexture("clips/checker.mov")
    val scene = Scene(
      camera = Camera.Default,
      objects = List(
        Cube(videoTexture = Some(videoTexture))
      ),
      lights = List(
        Directional((1f, -1f, 0f))
      )
    )

    val sceneConfig = convert(scene).scene
    val specs = sceneConfig.objectSpecs.get

    specs should have length 1
    specs(0).texture shouldBe None
    specs(0).videoTexture shouldBe Some(videoTexture)

  it should "create a scene with an environment-map video" in:
    val envMapVideo = EnvMapVideo(
      "clips/panorama.mov",
      VideoPlayback(timeMapping = TProgress, repeat = PingPong, fpsOverride = Some(24.0))
    )
    val scene = Scene(
      camera = Camera.Default,
      objects = List(Sphere(Material.Chrome)),
      envMapVideo = Some(envMapVideo)
    )

    val configs = convert(scene)

    configs.envMap shouldBe None
    configs.envMapVideo shouldBe Some(envMapVideo)

  it should "reject scenes that set both static and video environment maps" in:
    val scene = Scene(
      camera = Camera.Default,
      objects = List(Sphere(Material.Chrome)),
      envMap = Some("sky.hdr"),
      envMapVideo = Some(EnvMapVideo("clips/panorama.mov"))
    )

    an[IllegalArgumentException] shouldBe thrownBy(convert(scene))

  it should "create a complex scene with mixed object types" in:
    val scene = Scene(
      camera = Camera((5f, 5f, 5f), (0f, 0f, 0f)),
      objects = List(
        // Glass sphere
        Sphere((0f, 0f, 0f), Material.Glass, 1.0f),
        // Gold cube
        Cube((3f, 0f, 0f), Material.Gold, 1.2f),
        // Chrome Menger sponge
        Sponge(Vec3(-3f, 0f, 0f), VolumeFilling, level = 2f, Material.Chrome, size = 1.5f),
        // Copper surface-unfolding sponge
        Sponge(Vec3(0f, 3f, 0f), SurfaceUnfolding, level = 2.5f, Material.Copper)
      ),
      lights = List(
        Directional((1f, -1f, -1f), 1.0f, Color.White),
        Point((5f, 5f, 5f), 2.0f, Color("#FFFFCC")),
        Point((-5f, 5f, 5f), 1.5f, Color("#CCFFFF"))
      ),
      caustics = Caustics.HighQuality
    )

    val converted = convert(scene)
    val sceneConfig = converted.scene
    val cameraConfig = converted.camera

    // Verify all objects are present
    val specs = sceneConfig.objectSpecs.get
    specs should have length 4

    // Verify camera setup
    cameraConfig.position.x shouldBe 5f
    cameraConfig.position.y shouldBe 5f
    cameraConfig.position.z shouldBe 5f

    // Verify lights are configured
    scene.lights should have length 3

    // Verify caustics
    scene.caustics.get.photonsPerIteration shouldBe 500000

  "DSL to rendering pipeline" should "produce valid ObjectSpec for all scene objects" in:
    val objects = List(
      Sphere(Material.Glass),
      Cube(Material.Gold),
      Sponge(VolumeFilling, level = 2f),
      Sphere(color = Some(Color("#FF0000"))),
      Cube(texture = Some("wood.png"))
    )

    val scene = Scene(
      camera = Camera.Default,
      objects = objects,
      lights = List(Directional((1f, -1f, -1f)))
    )

    val sceneConfig = convert(scene).scene
    val specs = sceneConfig.objectSpecs.get

    // All objects should be converted
    specs should have length objects.length

    // All specs should be valid
    specs.foreach { spec =>
      spec.objectType should not be empty
      spec.size should be > 0f
    }

  "DSL render settings" should "flow through SceneConverter when set" in:
    val fallbackCaustics = Caustics.Disabled.toCausticsConfig
    val scene = Scene(
      camera = Camera.Default,
      objects = List(Sphere(Material.Chrome)),
      lights = List(Directional((1f, -1f, -1f))),
      render = Some(RenderSettings(shadows = true, antialiasing = true))
    )
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.render shouldBe defined
    result.render.get.shadows shouldBe true
    result.render.get.antialiasing shouldBe true

  it should "be None in SceneConfigs when not set" in:
    val fallbackCaustics = Caustics.Disabled.toCausticsConfig
    val scene = Scene(
      camera = Camera.Default,
      objects = List(Sphere(Material.Chrome)),
      lights = List(Directional((1f, -1f, -1f)))
    )
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.render shouldBe None

  it should "flow HighQuality preset through correctly" in:
    val fallbackCaustics = Caustics.Disabled.toCausticsConfig
    val scene = Scene(
      camera = Camera.Default,
      objects = List(Sphere(Material.Chrome)),
      lights = List(Directional((1f, -1f, -1f))),
      render = Some(RenderSettings.HighQuality)
    )
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.render shouldBe Some(RenderSettings.HighQuality.toRenderConfig)
