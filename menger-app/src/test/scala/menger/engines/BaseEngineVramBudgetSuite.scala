package menger.engines

import java.util.concurrent.atomic.AtomicReference

import scala.util.Failure
import scala.util.Success
import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.common.Light
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.common.Vector
import menger.dsl.ResourceLimits
import menger.engines.scene.HostHeapBudget
import menger.engines.scene.VramBudget
import menger.failure.Fail
import menger.objects.SpongeByVolume
import menger.optix.CameraState
import menger.optix.SceneConfigurator
import menger.spec.ObjectSpec
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// menger#179: the engine judges a scene's triangles against what the free GPU memory takes, and
// against the fixed ResourceLimits budget where it cannot read the memory. The GPU is played by
// `gpuMemory`, so none of this touches a device.
class BaseEngineVramBudgetSuite extends AnyFlatSpec with Matchers with MockFactory:

  given ProfilingConfig = ProfilingConfig.disabled

  private val MiB = 1024L * 1024
  private val GiB = 1024 * MiB
  private val volume5 = ObjectSpec(objectType = "sponge-volume", level = Some(5f))
  private val volume4 = ObjectSpec(objectType = "sponge-volume", level = Some(4f))
  private val volume5Triangles = SpongeByVolume.triangleCount(5)

  private def occupiedBy(triangles: Long): Long = triangles * VramBudget.BytesPerTriangle

  "BaseEngine.withinBudget" should "keep the fixed budget where the free GPU memory is unknown" in:
    val engine = TestEngine(None)
    engine.check(List(volume5)).isSuccess shouldBe true
    val refusal = engine.check(List(volume5, volume5)).failed.get.getMessage
    refusal should include("scene over budget")
    refusal should include(s"budget ${ResourceLimits.sceneMaxTriangles}")
    refusal should not include "GPU memory"

  it should "accept more triangles than the fixed budget on a GPU with more free memory" in:
    TestEngine(Some(24 * GiB)).check(List(volume5, volume5)).isSuccess shouldBe true

  it should "refuse a scene the free memory does not take, and say where the budget comes from" in:
    val engine = TestEngine(Some(2 * GiB))
    val refusal = engine.check(List(volume5)).failed.get.getMessage
    refusal should include("scene over budget")
    refusal should include(s"budget ${VramBudget.triangleBudget(VramBudget.GpuMemory(2 * GiB))}")
    refusal should include("free GPU memory: 2048 MiB free")
    refusal should include("sponge-volume fits at level 4")

  it should "accept the same scene once there is more free memory" in:
    val engine = TestEngine(Some(2 * GiB))
    engine.check(List(volume5)).isFailure shouldBe true
    engine.free.set(Some(3 * GiB))
    engine.check(List(volume5)).isSuccess shouldBe true

  it should "not blame the GPU memory for a refusal on another budget" in:
    // a sphere rendered with 256 accumulation frames is over the 128 renders per frame
    val refusal = TestEngine(Some(24 * GiB), accumulation = 256)
      .check(List(ObjectSpec(objectType = "sphere"))).failed.get.getMessage
    refusal should include("renders per frame")
    refusal should not include "GPU memory"

  it should "count the scene it replaces as free: a reload of the same scene fits" in:
    val engine = TestEngine(Some(2560L * MiB))
    engine.build(List(volume5))(Success(())).isSuccess shouldBe true
    // the scene now occupies its triangles: the card reports that much less free
    engine.free.set(Some(2560L * MiB - occupiedBy(volume5Triangles)))
    engine.check(List(volume5)).isSuccess shouldBe true

  it should "be a pure query: a scene that passed but was never built is not resident" in:
    val engine = TestEngine(Some(2560L * MiB))
    engine.check(List(volume5)).isSuccess shouldBe true
    engine.free.set(Some(2560L * MiB - occupiedBy(volume5Triangles)))
    engine.check(List(volume5)).isFailure shouldBe true

  it should "not count a refused scene as resident, it was never built" in:
    val engine = TestEngine(Some(2 * GiB))
    engine.build(List(volume5))(Success(())).isFailure shouldBe true
    engine.free.set(Some(1 * GiB))
    engine.check(List(volume5)).isFailure shouldBe true

  it should "not build a refused scene" in:
    val engine = TestEngine(Some(2 * GiB))
    engine.build(List(volume5))(fail("a scene over the budget must not be built"))
      .isFailure shouldBe true

  it should "count no scene as resident after a failed build" in:
    val engine = TestEngine(Some(2560L * MiB))
    engine.build(List(volume5))(Success(())).isSuccess shouldBe true
    engine.build(List(volume5))(Failure(IllegalStateException("upload failed")))
      .isFailure shouldBe true
    // the failed rebuild cleared the old scene; the card reports it as free, not credited twice
    engine.free.set(Some(2560L * MiB - occupiedBy(volume5Triangles)))
    engine.check(List(volume5)).isFailure shouldBe true

  it should "count no scene as resident after a build that throws" in:
    val engine = TestEngine(Some(2560L * MiB))
    engine.build(List(volume5))(Success(())).isSuccess shouldBe true
    engine.build(List(volume5))(Fail.internal("upload crashed"))
      .isFailure shouldBe true
    engine.free.set(Some(2560L * MiB - occupiedBy(volume5Triangles)))
    engine.check(List(volume5)).isFailure shouldBe true

  it should "count no scene as resident once the scene is cleared" in:
    val renderer = mock[OptiXRenderer]
    (renderer.clearAllInstances _).expects().once()
    val engine = TestEngine(Some(2560L * MiB))
    engine.build(List(volume5))(Success(())).isSuccess shouldBe true
    engine.clear(renderer)
    engine.free.set(Some(2560L * MiB - occupiedBy(volume5Triangles)))
    engine.check(List(volume5)).isFailure shouldBe true

  it should "refuse a scene the JVM heap does not take, however much GPU memory is free" in:
    val engine = TestEngine(Some(24 * GiB), heap = 4 * GiB)
    val refusal = engine.check(List(volume5, volume5)).failed.get.getMessage
    refusal should include("scene over budget")
    refusal should include(s"budget ${HostHeapBudget.triangleBudget(4 * GiB)}")
    refusal should include("JVM heap: 4096 MiB")
    refusal should not include "GPU memory"

  it should "accept the same scene with the heap for it" in:
    TestEngine(Some(24 * GiB), heap = 64 * GiB).check(List(volume5, volume5))
      .isSuccess shouldBe true

  it should "name the GPU memory when that is the smaller bound" in:
    val refusal = TestEngine(Some(2 * GiB), heap = 64 * GiB)
      .check(List(volume5)).failed.get.getMessage
    refusal should include("free GPU memory")
    refusal should not include "JVM heap"

  it should "reserve the first render's buffers until the renderer has rendered" in:
    val engine = TestEngine(Some(3 * GiB), firstRender = true)
    engine.check(List(volume5)).isFailure shouldBe true
    TestEngine(Some(3 * GiB)).check(List(volume5)).isSuccess shouldBe true

  private final class TestEngine(
    initialFree: Option[Long], accumulation: Int = 1, firstRender: Boolean = false,
    heap: Long = 1024 * GiB
  ) extends BaseEngine(64) with LiveReload:
    val free = AtomicReference(initialFree)
    private val eye = Vector[3](0f, 0f, 3f)
    override protected def sceneConfigurator: SceneConfigurator =
      SceneConfigurator(eye, Vector[3](0f, 0f, 0f), Vector[3](0f, 1f, 0f), Array.empty[Light])
    override protected def cameraState: CameraState =
      CameraState(eye, Vector[3](0f, 0f, 0f), Vector[3](0f, 1f, 0f))
    override protected def renderConfig: RenderConfig = RenderConfig.Default
    override protected def textureDir: String = "."
    override protected def eventsFor: Option[java.io.File] = None
    override protected def accumulationFrames: Int = accumulation
    override protected def freeGpuMemoryBytes: Option[Long] = free.get
    override protected def firstRenderPending: Boolean = firstRender
    override protected def maxHeapBytes: Long = heap

    def check(specs: List[ObjectSpec]): Try[Unit] = withinBudget(specs)
    def build(specs: List[ObjectSpec])(built: => Try[Unit]): Try[Unit] = building(specs)(built)
    def clear(renderer: OptiXRenderer): Unit = clearScene(renderer)
