package menger.dsl

import scala.language.implicitConversions

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneSuite extends AnyFlatSpec with Matchers:

  "Scene" should "be constructible with camera and single object" in:
    val sphere = Sphere(Material.Glass)
    val scene = Scene(Camera.Default, sphere)
    scene.camera shouldBe Camera.Default
    scene.objects should have length 1
    scene.objects.head shouldBe sphere
    scene.lights shouldBe empty
    scene.caustics shouldBe None

  it should "be constructible with camera and objects list" in:
    val sphere = Sphere(Vec3(-2f, 0f, 0f), Material.Glass)
    val cube = Cube(Vec3(2f, 0f, 0f), Material.Gold)
    val scene = Scene(Camera.Default, List(sphere, cube))
    scene.objects should have length 2

  it should "be constructible with camera, object, and lights" in:
    val sphere = Sphere(Material.Glass)
    val light = Directional(Vec3(1f, -1f, -1f))
    val scene = Scene(Camera.Default, sphere, List(light))
    scene.lights should have length 1

  it should "be constructible with camera, objects, and lights" in:
    val sphere = Sphere(Material.Glass)
    val cube = Cube(Material.Gold)
    val light1 = Directional(Vec3(1f, -1f, -1f))
    val light2 = Point(Vec3(0f, 5f, 0f))
    val scene = Scene(Camera.Default, List(sphere, cube), List(light1, light2))
    scene.objects should have length 2
    scene.lights should have length 2

  it should "be constructible with camera, objects, lights, and caustics" in:
    val sphere = Sphere(Material.Glass)
    val light = Directional(Vec3(1f, -1f, -1f))
    val caustics = Caustics.Default
    val scene = Scene(Camera.Default, List(sphere), List(light), caustics)
    scene.caustics shouldBe Some(caustics)

  it should "require at least one object" in:
    an[IllegalArgumentException] should be thrownBy Scene(Camera.Default, List.empty)

  "Scene.toSceneConfig" should "convert objects to SceneConfig correctly" in:
    val sphere = Sphere(Vec3(1f, 2f, 3f), Material.Glass, 0.5f)
    val cube = Cube(Vec3(-2f, 0f, 0f), Material.Copper, 1.5f)
    val scene = Scene(Camera.Default, List(sphere, cube))
    val config = scene.toSceneConfig

    config.isMultiObject shouldBe true
    config.objectSpecs shouldBe defined
    val specs = config.objectSpecs.get
    specs should have length 2

    specs(0).objectType shouldBe "sphere"
    specs(0).x shouldBe 1f
    specs(0).y shouldBe 2f
    specs(0).z shouldBe 3f
    specs(0).size shouldBe 0.5f

    specs(1).objectType shouldBe "cube"
    specs(1).x shouldBe -2f
    specs(1).size shouldBe 1.5f

  "Scene.toCameraConfig" should "convert camera correctly" in:
    val camera = new Camera(Vec3(1f, 2f, 3f), Vec3(4f, 5f, 6f))
    val sphere = Sphere(Material.Glass)
    val scene = Scene(camera, sphere)
    val cameraConfig = scene.toCameraConfig

    cameraConfig.position.x shouldBe 1f
    cameraConfig.position.y shouldBe 2f
    cameraConfig.position.z shouldBe 3f
    cameraConfig.lookAt.x shouldBe 4f
    cameraConfig.lookAt.y shouldBe 5f
    cameraConfig.lookAt.z shouldBe 6f

  "Scene with sponges" should "convert correctly" in:
    val sponge = Sponge(VolumeFilling, level = 2f, Material.Chrome)
    val scene = Scene(Camera.Default, List(sponge))
    val config = scene.toSceneConfig

    val specs = config.objectSpecs.get
    specs should have length 1
    specs(0).objectType shouldBe "sponge-volume"
    specs(0).level shouldBe Some(2f)
