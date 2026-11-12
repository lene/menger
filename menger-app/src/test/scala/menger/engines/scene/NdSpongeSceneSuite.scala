package menger.engines.scene

import scala.collection.mutable.ArrayBuffer

import io.github.lene.optix.OptiXRenderer
import menger.cli.ObjectSpecParser
import menger.common.Material
import menger.common.ProfilingConfig
import menger.common.TriangleMeshData
import menger.common.Vector
import menger.dsl.HypercubeSponge
import menger.dsl.ResourceLimits
import menger.dsl.TesseractSpongeType
import menger.engines.GeometryRegistry
import menger.geometry.MengerRenderer
import menger.objects.higher_d.IndexedMeshN
import menger.objects.higher_d.NdSpongeVariant
import menger.spec.ObjectSpec
import menger.spec.ObjectType
import menger.spec.PlaneName
import menger.spec.PlaneRotation
import menger.spec.ProjectionSpec
import org.scalatest.BeforeAndAfterEach
import org.scalatest.Outcome
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** The nD Menger sponge through the engines (menger#166): CPU triangles through the chain, the
  * memoised unprojected mesh, fractional levels, and the builder's choice of instances. */
class NdSpongeSceneSuite extends AnyFlatSpec with Matchers with BeforeAndAfterEach:

  given ProfilingConfig = ProfilingConfig.disabled

  private val Size = 1.2f
  private val MaxInstances = 64
  private val RoundingFactor = 10000f
  private val TrianglesPerFace = ResourceLimits.TrianglesPerFace.toInt
  private val Volume = "n-sponge-volume"
  private val Surface = "n-sponge-surface"

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var rendererOpt: Option[MengerRenderer] = None

  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  private def renderer: MengerRenderer =
    rendererOpt.getOrElse(throw new IllegalStateException("Renderer not initialized"))

  // only the builder tests need a renderer: the others never call it
  private def needsRenderer(test: NoArgTest): Boolean = test.name.startsWith("The builder")

  override def withFixture(test: NoArgTest): Outcome =
    if !needsRenderer(test) || rendererOpt.isDefined then super.withFixture(test)
    else cancel("OptiX native library not available")

  override def beforeEach(): Unit =
    super.beforeEach()
    try
      val _ = OptiXRenderer.isLibraryLoaded
      val r = MengerRenderer()
      if r.initialize(MaxInstances) then
        r.setCamera(Vector[3](0f, 0.5f, 4f), Vector[3](0f, 0f, 0f), Vector[3](0f, 1f, 0f), 45f)
        rendererOpt = Some(r)
    catch case _: Throwable => ()

  override def afterEach(): Unit =
    try rendererOpt.foreach(_.dispose())
    finally
      rendererOpt = None
      super.afterEach()

  private def turn(plane: String, degrees: Float) = PlaneRotation(PlaneName(plane), degrees)

  private def sponge(
    objectType: String, n: Int, level: Float, k: Option[Int] = None,
    material: Option[Material] = None
  ): ObjectSpec =
    ObjectSpec(
      objectType, size = Size, level = Some(level), dimension = Some(n),
      maxCentreCoordinates = k, projection = Some(ProjectionSpec(n)), material = material
    )

  private def cube(n: Int): ObjectSpec =
    ObjectSpec("n-cube", size = Size, dimension = Some(n), projection = Some(ProjectionSpec(n)))

  private def triangleCount(mesh: TriangleMeshData): Int = mesh.indices.length / 3

  /** Each face as the set of its rounded corner positions: the order of faces and corners is
    * not the geometry. */
  private def faceSet(mesh: TriangleMeshData): Set[Set[Seq[Int]]] =
    val corners = mesh.vertices.grouped(mesh.vertexStride).map(vertex =>
      vertex.take(3).toSeq.map(c => math.round(c * RoundingFactor))
    ).toSeq
    corners.grouped(4).map(_.toSet).toSet

  // ── triangles ────────────────────────────────────────────────────────────────────────

  "A 5D surface sponge of level 2" should "have 28,160 faces, two triangles each" in:
    val spec = sponge(Surface, 5, 2f)
    NdSpongeScene.mesh(spec, holeCaps = false).faceCount shouldBe 28160
    triangleCount(MeshFactory.create(spec)) shouldBe TrianglesPerFace * 28160

  "A sponge of level 0" should "be the n-cube, with the same view and size" in:
    for
      objectType <- Seq(Volume, Surface)
      n <- 4 to 8
    do
      withClue(s"$objectType, $n") {
        faceSet(MeshFactory.create(sponge(objectType, n, 0f))) shouldBe
          faceSet(MeshFactory.create(cube(n)))
      }

  "A 6D volume sponge with k = 4 (rule B)" should "have the prototype's 33,600 faces at level 1" in:
    val spec = sponge(Volume, 6, 1f, k = Some(4))
    NdSpongeScene.mesh(spec, holeCaps = false).faceCount shouldBe 33600
    triangleCount(MeshFactory.create(spec)) shouldBe TrianglesPerFace * 33600

  it should "differ from rule A" in:
    NdSpongeScene.mesh(sponge(Volume, 6, 1f), holeCaps = false).faceCount shouldBe 30720

  "A sponge in another view and scale" should
    "differ from the default and equal the n-cube in that view and scale at level 0" in:
      val view = ProjectionSpec(5, planes = List(turn("xv", 30f), turn("zw", 20f)))
      val scale = Some(IndexedSeq(1.2f, 1f, 0.8f, 1f, 0.5f))
      for objectType <- Seq(Volume, Surface) do
        val turned = sponge(objectType, 5, 0f).copy(projection = Some(view), scale = scale)
        val turnedCube = cube(5).copy(projection = Some(view), scale = scale)
        withClue(objectType) {
          faceSet(MeshFactory.create(turned)) shouldBe faceSet(MeshFactory.create(turnedCube))
          faceSet(MeshFactory.create(turned)) should not be
            faceSet(MeshFactory.create(sponge(objectType, 5, 0f)))
          val turnedLevel1 = MeshFactory.create(turned.copy(level = Some(1f)))
          faceSet(turnedLevel1) should not be
            faceSet(MeshFactory.create(sponge(objectType, 5, 1f)))
          faceSet(turnedLevel1) should not be
            faceSet(MeshFactory.create(turned.copy(level = Some(1f), scale = None)))
        }

  // ── the memo ─────────────────────────────────────────────────────────────────────────

  "The unprojected sponge" should "be built once for views, sizes and scales that differ" in:
    val spec = sponge(Surface, 5, 1f)
    val first = NdSpongeScene.mesh(spec, holeCaps = false)
    val turned = spec.copy(projection = Some(ProjectionSpec(6)), size = 2f, x = 3f)
    val scaled = spec.copy(scale = Some(IndexedSeq(1f, 1f, 1f, 1f, 0.5f)))
    NdSpongeScene.mesh(turned, holeCaps = false) should be theSameInstanceAs first
    NdSpongeScene.mesh(scaled, holeCaps = false) should be theSameInstanceAs first

  it should "be shared between a level and the fractional levels above it" in:
    NdSpongeScene.mesh(sponge(Surface, 5, 1.4f), holeCaps = false) should be theSameInstanceAs
      NdSpongeScene.mesh(sponge(Surface, 5, 1f), holeCaps = false)

  it should "be built again for another variant, dimension, level, k or for the caps" in:
    val base = sponge(Volume, 5, 1f)
    val first = NdSpongeScene.mesh(base, holeCaps = false)
    NdSpongeScene.mesh(base.copy(objectType = Surface), holeCaps = false) should not be
      theSameInstanceAs(first)
    NdSpongeScene.mesh(sponge(Volume, 6, 1f), holeCaps = false) should not be theSameInstanceAs(first)
    NdSpongeScene.mesh(sponge(Volume, 5, 2f), holeCaps = false) should not be theSameInstanceAs(first)
    NdSpongeScene.mesh(base.copy(maxCentreCoordinates = Some(2)), holeCaps = false) should not be
      theSameInstanceAs(first)
    NdSpongeScene.mesh(base, holeCaps = true) should not be theSameInstanceAs(first)

  it should "hold the meshes of every frame of PenteractSpongeBreath, built once each" in:
    val duration = examples.dsl.PenteractSpongeBreath.durationSeconds
    val frames = (0 until 64).map(i =>
      examples.dsl.PenteractSpongeBreath.scene(i * duration / 32f).objects.head.toObjectSpec
    )
    // what the builder draws: a whole level, or the level above plus the caps of the level
    def meshes(spec: ObjectSpec): Seq[((Int, Boolean), IndexedMeshN)] =
      val level = spec.level.get
      if level == level.floor then Seq((level.toInt, false) -> NdSpongeScene.mesh(spec, false))
      else
        Seq(
          ((level.floor + 1).toInt, false) ->
            NdSpongeScene.mesh(spec.copy(level = Some(level.floor + 1)), holeCaps = false),
          (level.floor.toInt, true) ->
            NdSpongeScene.mesh(spec.copy(level = Some(level.floor)), holeCaps = true)
        )
    val byKey = frames.flatMap(meshes).groupMap(_._1)(_._2)
    byKey.keySet should contain allOf ((0, false), (1, false), (2, false), (0, true), (1, true))
    byKey.foreach { (key, instances) =>
      withClue(key) { instances.foreach(_ should be theSameInstanceAs instances.head) }
    }

  it should "weigh what it retains in faces, within the scene's triangle budget" in:
    (4 to 8).foreach(n => NdSpongeScene.mesh(sponge(Surface, n, 1f), holeCaps = false))
    NdSpongeScene.retainedFaces should be <= ResourceLimits.sceneMaxTriangles

  // ── fractional levels ────────────────────────────────────────────────────────────────

  "The hole caps of an opaque fractional sponge" should
    "have as many faces as the lower level, in a second plan beside the upper level" in:
      val spec = sponge(Surface, 5, 1.4f, material = Some(Material.Matte))
      val upper = MeshFactory.createUpload(spec.copy(level = Some(2f)))
      val caps = MeshFactory.createUpload(spec.copy(level = Some(1f)), holeCaps = true)
      (upper, caps) match
        case (MeshUploadPlan.Cpu(upperMesh), MeshUploadPlan.Cpu(capMesh)) =>
          triangleCount(upperMesh) shouldBe TrianglesPerFace * 28160
          triangleCount(capMesh) shouldBe TrianglesPerFace * 1600
        case other => fail(s"CPU plans expected, got $other")
      TriangleMeshSceneBuilder.holeCapsCoverage(1.4f) shouldBe 0.6f +- 1e-5f

  "The builder" should "draw an opaque fractional sponge as level 2 plus the caps of level 1" in:
    val instances = ArrayBuffer.empty[Int]
    val caps = ArrayBuffer.empty[Int]
    val builder = TriangleMeshSceneBuilder(
      ".",
      holeCapsRecorder = (_, id) => { caps += id.value; () },
      instanceRecorder = (_, id) => { instances += id.value; () }
    )
    val spec = sponge(Surface, 5, 1.4f, material = Some(Material.Matte))
    builder.calculateInstanceCount(List(spec)) shouldBe 2L
    builder.validateAndBuild(List(spec), renderer, MaxInstances).get
    instances should have size 2
    caps should have size 1
    renderer.levelMixCount shouldBe 0

  it should "draw a glass fractional sponge as two whole levels, mixed" in:
    val instances = ArrayBuffer.empty[Int]
    val caps = ArrayBuffer.empty[Int]
    val builder = TriangleMeshSceneBuilder(
      ".",
      holeCapsRecorder = (_, id) => { caps += id.value; () },
      instanceRecorder = (_, id) => { instances += id.value; () }
    )
    val spec = sponge(Surface, 5, 1.4f, material = Some(Material.Glass))
    builder.calculateInstanceCount(List(spec)) shouldBe 2L
    builder.validateAndBuild(List(spec), renderer, MaxInstances).get
    instances should have size 2
    caps shouldBe empty
    renderer.levelMixCount shouldBe 1

  it should "draw a whole level as one instance" in:
    val spec = sponge(Volume, 5, 1f, material = Some(Material.Matte))
    val builder = TriangleMeshSceneBuilder(".")
    builder.calculateInstanceCount(List(spec)) shouldBe 1L
    builder.validateAndBuild(List(spec), renderer, MaxInstances).get
    renderer.levelMixCount shouldBe 0

  // ── routing ──────────────────────────────────────────────────────────────────────────

  "The n-sponge types" should "be triangle meshes on neither 4D fast path nor with edge tubes" in:
    for objectType <- Seq(Volume, Surface) do
      ObjectType.isValid(objectType) shouldBe true
      ObjectType.isNdSponge(objectType) shouldBe true
      ObjectType.isTriangleMesh(objectType) shouldBe true
      ObjectType.isProjected4D(objectType) shouldBe false
      ObjectType.isEdgeTubeMesh(objectType) shouldBe false
      ObjectType.is4DSponge(objectType) shouldBe false

  they should "take the triangle mesh builder" in:
    GeometryRegistry.builderFor(List(sponge(Surface, 5, 1f))).get shouldBe
      a[TriangleMeshSceneBuilder]

  "The command line" should "refuse both n-sponge types and point to the DSL" in:
    for objectType <- Seq(Volume, Surface) do
      val refusal = ObjectSpecParser.parse(s"type=$objectType:level=1")
      refusal.isLeft shouldBe true
      refusal.left.toOption.get should (
        include("DSL") and include("type=tesseract-sponge-volume") and
          include("type=tesseract-sponge-surface")
      )

  "NdSpongeScene.variantOf" should "map the two types to the variants" in:
    NdSpongeScene.variantOf(sponge(Volume, 5, 1f)) shouldBe NdSpongeVariant.Volume
    NdSpongeScene.variantOf(sponge(Surface, 5, 1f)) shouldBe NdSpongeVariant.Surface

  "The mapping between rule, DSL type and object type" should "be one, in every direction" in:
    for (spongeType, variant) <- Seq(
      TesseractSpongeType.VolumeRemoving -> NdSpongeVariant.Volume,
      TesseractSpongeType.SurfaceSubdividing -> NdSpongeVariant.Surface
    ) do
      HypercubeSponge.variantOf(spongeType) shouldBe variant
      NdSpongeVariant.fromObjectType(variant.objectTypeName) shouldBe Some(variant)
      HypercubeSponge(5, spongeType, 0f).toObjectSpec.objectType shouldBe variant.objectTypeName
    ObjectType.NSPONGE_TYPES shouldBe NdSpongeVariant.values.map(_.objectTypeName).toSet
    NdSpongeVariant.fromObjectType("tesseract-sponge-volume") shouldBe None

  "An upload plan without triangles" should "be left out of the scene" in:
    val none = TriangleMeshData(Array.emptyFloatArray, Array.emptyIntArray, 8)
    val some = TriangleMeshData(Array.fill(24)(0f), Array(0, 1, 2), 8)
    TriangleMeshSceneBuilder.isEmptyCpuMesh(MeshUploadPlan.Cpu(none)) shouldBe true
    TriangleMeshSceneBuilder.isEmptyCpuMesh(MeshUploadPlan.Cpu(some)) shouldBe false

  "A sponge with every face clipped" should "project to a mesh without triangles" in:
    val mesh = NdSpongeScene.mesh(sponge(Surface, 5, 1f), holeCaps = false)
    val none = mesh.toTriangleMesh(Size, IndexedSeq.fill(5)(1f), _ => None, Vector.Zero[3])
    none.indices shouldBe empty
