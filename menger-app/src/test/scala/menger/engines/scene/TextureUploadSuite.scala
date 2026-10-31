package menger.engines.scene

import java.nio.file.Files

import io.github.lene.optix.OptiXRenderer
import org.scalatest.BeforeAndAfterEach
import org.scalatest.Outcome
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// menger#103: TextureManager.uploadFile is the one place a file upload's failures are caught;
// the environment map and HDR texture paths each had their own copy of the catch.
class TextureUploadSuite extends AnyFlatSpec with Matchers with BeforeAndAfterEach:

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
      if r.initialize(1) then rendererOpt = Some(r)
    catch case _: Throwable => ()

  override def afterEach(): Unit =
    try rendererOpt.foreach(_.dispose())
    finally
      rendererOpt = None
      super.afterEach()

  private val textureDir = Files.createTempDirectory("texture-upload-").toString

  private def upload(path: String): Either[String, ?] =
    rendererOpt.map(TextureManager.uploadFile(_, textureDir, path)).getOrElse(fail("no renderer"))

  "TextureManager.uploadFile" should "refuse a path outside the texture directory" in:
    upload("../outside.hdr").left.map(_.startsWith("cannot resolve:")) shouldBe Left(true)

  it should "refuse an absolute path" in:
    upload("/etc/passwd").left.map(_.startsWith("cannot resolve:")) shouldBe Left(true)

  it should "report a file that does not exist as an upload failure, not throw" in:
    upload("missing.hdr").left.map(_.startsWith("upload failed:")) shouldBe Left(true)
