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

  it should "default render to None when scene has no render field" in:
    val scene = Scene(Camera.Default, Sphere(Material.Glass))
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.render shouldBe None

  it should "include render config when scene has render set" in:
    val scene = Scene(Camera.Default, Sphere(Material.Glass)).copy(render = Some(RenderSettings(shadows = true)))
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.render shouldBe defined
    result.render.get.shadows shouldBe true

  it should "default envMap to None when scene has no envMap set" in:
    val scene = Scene(Camera.Default, Sphere(Material.Glass))
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.envMap shouldBe None

  it should "propagate envMap from scene to SceneConfigs" in:
    val scene = Scene(Camera.Default, Sphere(Material.Glass)).copy(envMap = Some("panorama.hdr"))
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.envMap shouldBe Some("panorama.hdr")

  it should "default toneMappingOperator to 0 (none) when scene has no toneMapping set" in:
    val scene = Scene(Camera.Default, Sphere(Material.Glass))
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.toneMappingOperator shouldBe 0
    result.toneMappingExposure shouldBe 1.0f

  it should "propagate Reinhard toneMapping to operator=1 and correct exposure" in:
    val scene = Scene(Camera.Default, Sphere(Material.Glass)).copy(toneMapping = ToneMapping.Reinhard(2.0f))
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.toneMappingOperator shouldBe 1
    result.toneMappingExposure shouldBe 2.0f

  it should "propagate ACES toneMapping to operator=2 and correct exposure" in:
    val scene = Scene(Camera.Default, Sphere(Material.Glass)).copy(toneMapping = ToneMapping.ACES(1.5f))
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.toneMappingOperator shouldBe 2
    result.toneMappingExposure shouldBe 1.5f

  it should "handle objects without materials" in:
    val sphere = Sphere()
    val cube = Cube()
    val sponge = Sponge(SpongeType.SurfaceUnfolding, 2f)
    val tesseract = Tesseract()
    val tesseractSponge = TesseractSponge(TesseractSpongeType.SurfaceSubdividing, level = 1f)
    val scene = Scene(Camera.Default, List(sphere, cube, sponge, tesseract, tesseractSponge))
    noException should be thrownBy SceneConverter.convert(scene, fallbackCaustics)
