package menger.engines.scene

import menger.ObjectSpec
import menger.common.ProfilingConfig
import menger.engines.GeometryRegistry
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.{Failure, Success}

/** Determinism + JNI fault-injection fitness function (T7, Sprint 32).
  *
  * Verifies:
  * - `InstanceId.fromNative(-1)` propagates the expected error
  * - SceneBuilder `Try { }` wraps the error as `Failure`
  * - Builders handle null/invalid inputs by returning Failure
  */
class SceneBuilderFaultInjectionSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  "InstanceId.fromNative" should "throw on -1 with a descriptive message" in:
    val ex = intercept[RuntimeException]:
      InstanceId.fromNative(-1, "test operation")
    ex.getMessage should include("test operation")

  it should "return a valid InstanceId for non-negative values" in:
    InstanceId.raw(InstanceId.fromNative(0, "test")) shouldBe 0
    InstanceId.raw(InstanceId.fromNative(42, "test")) shouldBe 42

  it should "throw on any negative value" in:
    val ex = intercept[RuntimeException]:
      InstanceId.fromNative(-5, "sphere add")
    ex.getMessage should include("sphere add")

  "SceneBuilder.buildScene" should "return Failure on exception during build (null renderer)" in:
    val builder = SphereSceneBuilder()
    val spec = ObjectSpec(objectType = "sphere")
    builder.buildScene(List(spec), null, 100).isFailure shouldBe true

  it should "return Success for empty specs" in:
    SphereSceneBuilder().buildScene(List.empty, null, 100).isSuccess shouldBe true

  "ConeSceneBuilder" should "return Failure with null renderer" in:
    val builder = ConeSceneBuilder()
    val spec = ObjectSpec(objectType = "cone")
    builder.buildScene(List(spec), null, 100).isFailure shouldBe true

  "CurveSceneBuilder" should "return Failure with null renderer" in:
    val builder = CurveSceneBuilder()
    val spec = ObjectSpec(objectType = "curve")
    builder.buildScene(List(spec), null, 100).isFailure shouldBe true

  "GeometryRegistry" should "propagate Failure when builder fails with null renderer" in:
    val spec = ObjectSpec(objectType = "cone")
    val builderOpt = GeometryRegistry.builderFor(List(spec))
    builderOpt.isDefined shouldBe true
    builderOpt.get.buildScene(List(spec), null, 100).isFailure shouldBe true

  "TriangleMeshSceneBuilder" should "return Failure with null renderer" in:
    val builder = TriangleMeshSceneBuilder(".")
    val spec = ObjectSpec(objectType = "cube")
    builder.buildScene(List(spec), null, 100).isFailure shouldBe true
