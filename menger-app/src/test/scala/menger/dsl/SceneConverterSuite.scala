package menger.dsl

import scala.language.implicitConversions

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneConverterSuite extends AnyFlatSpec with Matchers:

  private val fallbackCaustics = Caustics.Disabled.toCausticsConfig

  "SceneConverter.convert" should "convert a basic sphere scene" in:
    val scene = Scene(Camera.Default, Sphere(Material.Glass))
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.lights shouldBe empty
    result.background shouldBe None
    result.planes shouldBe empty
    result.caustics shouldBe fallbackCaustics

  it should "include caustics from the scene if set" in:
    val scene = Scene(Camera.Default, List(Sphere(Material.Glass)), List.empty, Caustics.Default)
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.caustics should not be fallbackCaustics

  it should "include lights from the scene" in:
    val light = Directional(Vec3(1f, -1f, -1f))
    val scene = Scene(Camera.Default, Sphere(Material.Glass), List(light))
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.lights should have length 1

  it should "include background color if set" in:
    val scene = Scene(Camera.Default, Sphere(Material.Glass)).copy(background = Some(Color(0.2f, 0.3f, 0.4f)))
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.background shouldBe defined

  it should "include planes if set" in:
    val plane = Plane(Y at -1f, Color(0.5f, 0.5f, 0.5f))
    val scene = Scene(Camera.Default, Sphere(Material.Glass)).copy(planes = List(plane))
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.planes should have length 1

  "SceneConverter.validateSceneMaterials" should "accept all object types without error" in:
    val sphere = Sphere(Material.Glass)
    val cube = Cube(Material.Gold)
    val sponge = Sponge(SpongeType.VolumeFilling, 2f, Material.Copper)
    val tesseract = Tesseract(material = Some(Material.Glass), edgeMaterial = Some(Material.Gold))
    val tesseractSponge = TesseractSponge(
      TesseractSpongeType.VolumeRemoving, level = 1f,
      material = Some(Material.Copper), edgeMaterial = Some(Material.Copper)
    )
    val scene = Scene(Camera.Default, List(sphere, cube, sponge, tesseract, tesseractSponge))
    noException should be thrownBy SceneConverter.convert(scene, fallbackCaustics)

  it should "handle objects without materials" in:
    val sphere = Sphere()
    val cube = Cube()
    val sponge = Sponge(SpongeType.SurfaceUnfolding, 2f)
    val tesseract = Tesseract()
    val tesseractSponge = TesseractSponge(TesseractSpongeType.SurfaceSubdividing, level = 1f)
    val scene = Scene(Camera.Default, List(sphere, cube, sponge, tesseract, tesseractSponge))
    noException should be thrownBy SceneConverter.convert(scene, fallbackCaustics)
