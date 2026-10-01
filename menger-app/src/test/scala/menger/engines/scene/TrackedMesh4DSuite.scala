package menger.engines.scene

import io.github.lene.optix.OptiXRenderer
import menger.ObjectRotation
import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.ImageSize
import menger.common.ProfilingConfig
import menger.common.Vector
import org.scalatest.BeforeAndAfterEach
import org.scalatest.Outcome
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** An animated fractional 4D sponge is updated in place (projection + hole-cap alpha) instead
  * of rebuilt every frame (usability review 2026-09, session 2, F52). The updated scene has to
  * render exactly like a fresh build. */
class TrackedMesh4DSuite extends AnyFlatSpec with Matchers with BeforeAndAfterEach:
  given ProfilingConfig = ProfilingConfig.disabled

  private val Size = ImageSize(200, 150)
  private val MaxInstances = 64

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var rendererOpt: Option[OptiXRenderer] = None

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def renderer: OptiXRenderer =
    rendererOpt.getOrElse(throw new IllegalStateException("Renderer not initialized"))

  override def withFixture(test: NoArgTest): Outcome =
    if rendererOpt.isDefined then super.withFixture(test)
    else cancel("OptiX native library not available")

  override def beforeEach(): Unit =
    super.beforeEach()
    try
      val _ = OptiXRenderer.isLibraryLoaded
      val r = new OptiXRenderer()
      if r.initialize(MaxInstances) then
        r.setCamera(Vector[3](0f, 0.5f, 4f), Vector[3](0f, 0f, 0f), Vector[3](0f, 1f, 0f), 45f)
        r.setLight(Vector[3](0.4f, -0.6f, -0.5f), 1f)
        rendererOpt = Some(r)
    catch case _: Throwable => ()

  override def afterEach(): Unit =
    try rendererOpt.foreach(_.dispose())
    finally
      rendererOpt = None
      super.afterEach()

  private def sponge(level: Float, rotXW: Float): ObjectSpec =
    ObjectSpec.parse(s"type=tesseract-sponge:level=$level:size=1.5:material=film")
      .fold(e => fail(e), identity)
      .copy(projection4D = Some(Projection4DSpec(rotXW = rotXW)))

  private def build(specs: List[ObjectSpec]): TrackedMesh4D.State =
    TrackedMesh4D.build(specs, renderer, ".", _ => MaxInstances).get
      .getOrElse(fail("the scene was not tracked"))

  "TrackedMesh4D.updateInPlace" should "render exactly like a fresh build" in:
    val before = List(sponge(1.2f, 20f))
    val after = List(sponge(1.8f, 35f))
    val state = build(before)
    state.capsInstancePerSpec.head shouldBe defined
    val first = renderer.render(Size)

    TrackedMesh4D.canUpdateInPlace(before, after) shouldBe true
    val updated = TrackedMesh4D.updateInPlace(state, after, renderer)
    updated.specs shouldBe after
    val moved = renderer.render(Size)

    renderer.clearAllInstances()
    build(after)
    val fresh = renderer.render(Size)

    moved should not equal first
    moved shouldEqual fresh

  it should "move and rotate the object like a fresh build" in:
    val before = List(sponge(1.2f, 20f))
    val after = List(
      sponge(1.5f, 20f).copy(x = 0.4f, y = -0.2f, rotation = ObjectRotation(0f, 30f, 0f))
    )
    val state = build(before)
    state.instancesPerSpec.head.size shouldBe 2  // level-2 mesh + level-1 hole caps
    val first = renderer.render(Size)

    TrackedMesh4D.canUpdateInPlace(before, after) shouldBe true
    val _ = TrackedMesh4D.updateInPlace(state, after, renderer)
    val moved = renderer.render(Size)

    renderer.clearAllInstances()
    build(after)
    val fresh = renderer.render(Size)

    moved should not equal first
    moved shouldEqual fresh
