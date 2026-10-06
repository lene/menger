package io.github.lene.optix

import menger.ObjectSpec
import menger.common.Material
import menger.common.ProfilingConfig
import menger.engines.scene.TesseractEdgeSceneBuilder
import menger.engines.scene.TriangleMeshSceneBuilder
import org.scalatest.BeforeAndAfterEach
import org.scalatest.Outcome
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** optix-jni 0.4.4 (#41): `setProjectedMesh` fails with -2 when the projected vertices can't be
  * read back, instead of registering untrusted geometry. Both builders that upload projected 4D
  * meshes must turn that into a failed build, which the engines report with the
  * FRAME-BUILD-FAILED marker (menger#54). The failure is injected through the 0.4.4 test hook. */
class ProjectedMeshFailureSuite extends AnyFlatSpec with Matchers with BeforeAndAfterEach:

  given ProfilingConfig = ProfilingConfig.disabled

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private var rendererOpt: Option[OptiXRenderer] = None

  override def withFixture(test: NoArgTest): Outcome =
    if rendererOpt.isDefined then super.withFixture(test)
    else cancel("OptiX native library not available")

  override def beforeEach(): Unit =
    super.beforeEach()
    try
      val _ = OptiXRenderer.isLibraryLoaded
      val r = new OptiXRenderer()
      if r.initialize() then rendererOpt = Some(r)
    catch case _: Throwable => ()

  override def afterEach(): Unit =
    try rendererOpt.foreach(_.dispose())
    finally
      rendererOpt = None
      super.afterEach()

  private def failingRenderer: OptiXRenderer =
    val r = rendererOpt.get
    r.failNextProjectionReadbackNative()
    r

  "TriangleMeshSceneBuilder" should "fail the build when the projection readback fails" in:
    val result = TriangleMeshSceneBuilder(".")
      .buildScene(List(ObjectSpec(objectType = "tesseract")), failingRenderer, 100)
    result.isFailure shouldBe true
    result.failed.get.getMessage should include("-2")

  "TesseractEdgeSceneBuilder" should "fail the build when the projection readback fails" in:
    // Faces (the projected mesh) are uploaded only when the spec has a face material.
    val spec = ObjectSpec(
      objectType = "tesseract", size = 1, edgeRadius = Some(0.02f), material = Some(Material.Gold)
    )
    // At most the renderer's capacity: a larger maxInstances reinitializes the native renderer,
    // which discards the armed failure hook.
    val renderer = failingRenderer
    val result = TesseractEdgeSceneBuilder(".")
      .buildScene(List(spec), renderer, renderer.maxInstances)
    result.isFailure shouldBe true
    result.failed.get.getMessage should include("-2")
