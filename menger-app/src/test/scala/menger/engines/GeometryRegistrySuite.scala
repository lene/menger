package menger.engines

import menger.ObjectSpec
import menger.common.ProfilingConfig
import menger.engines.scene.CubeSpongeSceneBuilder
import menger.engines.scene.SphereSceneBuilder
import menger.engines.scene.TesseractEdgeSceneBuilder
import menger.engines.scene.TriangleMeshSceneBuilder
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GeometryRegistrySuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  private def spec(t: String) = ObjectSpec(objectType = t, x = 0, y = 0, z = 0, size = 1)

  "GeometryRegistry.builderFor" should "return SphereSceneBuilder for all-sphere specs" in:
    val result = GeometryRegistry.builderFor(List(spec("sphere")))
    result shouldBe defined
    result.get shouldBe a [SphereSceneBuilder]

  it should "return SphereSceneBuilder for multiple sphere specs" in:
    val result = GeometryRegistry.builderFor(List(spec("sphere"), spec("sphere")))
    result shouldBe defined
    result.get shouldBe a [SphereSceneBuilder]

  it should "return CubeSpongeSceneBuilder for all-cube-sponge specs" in:
    val result = GeometryRegistry.builderFor(List(spec("cube-sponge")))
    result shouldBe defined
    result.get shouldBe a [CubeSpongeSceneBuilder]

  it should "return TriangleMeshSceneBuilder for cube specs" in:
    val result = GeometryRegistry.builderFor(List(spec("cube")))
    result shouldBe defined
    result.get shouldBe a [TriangleMeshSceneBuilder]

  it should "return TriangleMeshSceneBuilder for sponge-volume specs" in:
    val result = GeometryRegistry.builderFor(List(spec("sponge-volume")))
    result shouldBe defined
    result.get shouldBe a [TriangleMeshSceneBuilder]

  it should "return TriangleMeshSceneBuilder for sponge-recursive-ias specs" in:
    val result = GeometryRegistry.builderFor(List(spec("sponge-recursive-ias")))
    result shouldBe defined
    result.get shouldBe a [TriangleMeshSceneBuilder]

  it should "return TriangleMeshSceneBuilder for tesseract specs without edge rendering" in:
    val result = GeometryRegistry.builderFor(List(spec("tesseract")))
    result shouldBe defined
    result.get shouldBe a [TriangleMeshSceneBuilder]

  it should "return TesseractEdgeSceneBuilder for tesseract specs with edge rendering" in:
    val edgeSpec = ObjectSpec(
      objectType = "tesseract", x = 0, y = 0, z = 0, size = 1, edgeRadius = Some(0.02f)
    )
    val result = GeometryRegistry.builderFor(List(edgeSpec))
    result shouldBe defined
    result.get shouldBe a [TesseractEdgeSceneBuilder]

  it should "return None for mixed sphere + cube specs" in:
    val result = GeometryRegistry.builderFor(List(spec("sphere"), spec("cube")))
    result shouldBe None

  it should "return None for empty spec list" in:
    val result = GeometryRegistry.builderFor(List.empty)
    result shouldBe None
