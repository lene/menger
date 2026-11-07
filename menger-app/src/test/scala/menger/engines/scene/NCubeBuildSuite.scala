package menger.engines.scene

import scala.language.implicitConversions

import io.github.lene.optix.OptiXRenderer
import menger.common.Material
import menger.common.ProfilingConfig
import menger.common.Vector
import menger.spec.ObjectSpec
import menger.spec.PlaneName
import menger.spec.PlaneRotation
import menger.spec.ProjectionSpec
import menger.spec.ProjectionStage
import org.scalatest.BeforeAndAfterEach
import org.scalatest.Outcome
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** An n-cube built through `TesseractEdgeSceneBuilder.buildScene` on a real renderer
  * (menger#163): one cylinder instance per edge, one instance for the faces. Cancelled without
  * OptiX. */
class NCubeBuildSuite extends AnyFlatSpec with Matchers with BeforeAndAfterEach:
  given ProfilingConfig = ProfilingConfig.disabled

  private val MaxInstances = 4096
  private val PenteractEdges = 80

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
        rendererOpt = Some(r)
    catch case _: Throwable => ()

  override def afterEach(): Unit =
    try rendererOpt.foreach(_.dispose())
    finally
      rendererOpt = None
      super.afterEach()

  private def penteract(
    faces: Boolean, view: Option[ProjectionSpec] = None, size: Float = 1.2f
  ): ObjectSpec =
    ObjectSpec(
      "n-cube", size = size, dimension = Some(5), projection = view,
      material = Option.when(faces)(Material.Film),
      edgeRadius = Some(0.02f), edgeMaterial = Some(Material.Gold)
    )

  private def instancesAfterBuilding(spec: ObjectSpec): Int =
    TesseractEdgeSceneBuilder(".").validateAndBuild(List(spec), renderer, MaxInstances).get
    renderer.getInstanceCount()

  "Building an edge-rendered 5-cube" should "add one cylinder instance per edge" in:
    instancesAfterBuilding(penteract(faces = false)) shouldBe PenteractEdges

  it should "add one more instance for the faces when the cube has a face material" in:
    instancesAfterBuilding(penteract(faces = true)) shouldBe PenteractEdges + 1

  it should "keep every tube of a lifted, turned cube" in:
    val view = ProjectionSpec(7, planes = List(PlaneRotation(PlaneName("vt"), 40f)))
    instancesAfterBuilding(penteract(faces = true, Some(view))) shouldBe PenteractEdges + 1

  it should "leave out the tubes past a stage's eye" in:
    // Size 2, first stage eye 0.6: every edge with an endpoint at v = +1 is clipped, 32 remain.
    val view = ProjectionSpec(
      5, stages = List(ProjectionStage(0.6f, 0.3f), ProjectionStage(6f, 1.5f))
    )
    val clipped = 32
    instancesAfterBuilding(penteract(faces = true, Some(view), size = 2f)) shouldBe clipped + 1
