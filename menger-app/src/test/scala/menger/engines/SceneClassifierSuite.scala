package menger.engines

import menger.ObjectSpec
import menger.ProfilingConfig
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneClassifierSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  private def spec(t: String) = ObjectSpec(objectType = t, x = 0, y = 0, z = 0, size = 1)

  "SceneClassifier.isTriangleMeshType" should "return true for cube" in:
    SceneClassifier.isTriangleMeshType("cube") shouldBe true

  it should "return true for sponge types" in:
    SceneClassifier.isTriangleMeshType("sponge-volume") shouldBe true
    SceneClassifier.isTriangleMeshType("sponge-surface") shouldBe true

  it should "return false for sphere" in:
    SceneClassifier.isTriangleMeshType("sphere") shouldBe false

  "SceneClassifier.classify" should "classify all-sphere scene as Spheres" in:
    val specs = List(spec("sphere"), spec("sphere"))
    SceneClassifier.classify(specs) shouldBe a [SceneType.Spheres]

  it should "classify cube-sponge scene as CubeSponges" in:
    val specs = List(spec("cube-sponge"))
    SceneClassifier.classify(specs) shouldBe a [SceneType.CubeSponges]

  it should "classify all-cube scene as TriangleMeshes" in:
    val specs = List(spec("cube"), spec("cube"))
    SceneClassifier.classify(specs) shouldBe a [SceneType.TriangleMeshes]

  it should "classify sphere + cube as SimpleMixed" in:
    val specs = List(spec("sphere"), spec("cube"))
    SceneClassifier.classify(specs) shouldBe a [SceneType.SimpleMixed]

  it should "classify sphere + cube + sponge as ComplexMixed" in:
    val specs = List(spec("sphere"), spec("cube"), spec("sponge-volume"))
    SceneClassifier.classify(specs) shouldBe a [SceneType.ComplexMixed]

  "SceneClassifier.selectSceneBuilder" should "return SphereSceneBuilder for Spheres" in:
    val result = SceneClassifier.selectSceneBuilder(
      SceneType.Spheres(List(spec("sphere"))), textureDir = None
    )
    result shouldBe defined

  it should "return TesseractEdgeSceneBuilder for 4D projected specs with edge rendering" in:
    val edgeSpec = ObjectSpec(objectType = "tesseract", x = 0, y = 0, z = 0, size = 1, edgeRadius = Some(0.02f))
    val result = SceneClassifier.selectSceneBuilder(
      SceneType.TriangleMeshes(List(edgeSpec)), textureDir = None
    )
    result shouldBe defined
    result.get shouldBe a [menger.engines.scene.TesseractEdgeSceneBuilder]

  it should "return TriangleMeshSceneBuilder for 4D projected specs without edge rendering" in:
    val result = SceneClassifier.selectSceneBuilder(
      SceneType.TriangleMeshes(List(spec("tesseract"))), textureDir = None
    )
    result shouldBe defined
    result.get shouldBe a [menger.engines.scene.TriangleMeshSceneBuilder]

  it should "return CubeSpongeSceneBuilder for CubeSponges" in:
    val result = SceneClassifier.selectSceneBuilder(
      SceneType.CubeSponges(List(spec("cube-sponge"))), textureDir = None
    )
    result shouldBe defined
    result.get shouldBe a [menger.engines.scene.CubeSpongeSceneBuilder]

  it should "return None for SimpleMixed" in:
    val result = SceneClassifier.selectSceneBuilder(
      SceneType.SimpleMixed(List(spec("sphere")), "cube"), textureDir = None
    )
    result shouldBe None
