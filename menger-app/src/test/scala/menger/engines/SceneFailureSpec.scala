package menger.engines

import io.github.lene.optix.OptiXContextLostException
import io.github.lene.optix.OptiXException
import io.github.lene.optix.TextureUploadException
import menger.common.ConfigurationException
import menger.common.EnvironmentException
import menger.common.InternalException
import menger.common.NotYetImplementedException
import menger.common.UnknownGeometryException
import menger.common.ValidationException
import menger.engines.scene.AssetPaths
import menger.scene.LoadError
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** menger#103: a window's failures reached its seams as a Throwable here, a String there, and
  * a LoadError elsewhere, and a bad scene, a failing GPU and a bug read alike. One translation
  * classifies the exception; the seams (frame, reload, startup) take the result. */
class SceneFailureSpec extends AnyFlatSpec with Matchers:

  private def kindOf(cause: Throwable): String = SceneFailure.of(cause).kind

  // The native layer's own constructor takes the cause too.
  private def gpuFailure(message: String): OptiXException =
    new OptiXException(message, null) // scalafix:ok DisableSyntax.null

  "SceneFailure.of" should "call a rejected scene the scene's fault" in:
    kindOf(ValidationException("too many instances", "objectSpecs", 9999)) shouldBe "scene"
    kindOf(ConfigurationException("bad option")) shouldBe "scene"
    kindOf(IllegalArgumentException("requirement failed: Size must be positive")) shouldBe "scene"
    kindOf(UnsupportedOperationException("No builder for sponge")) shouldBe "scene"
    kindOf(AssetPaths.AssetPathException("escapes --texture-dir")) shouldBe "scene"

  it should "call a failure of the GPU, the native layer or the disk the environment's" in:
    kindOf(gpuFailure("illegal memory access")) shouldBe "environment"
    kindOf(new OptiXContextLostException("context lost")) shouldBe "environment"
    kindOf(TextureUploadException("cannot read texture")) shouldBe "environment"
    kindOf(java.io.IOException("disk full")) shouldBe "environment"

  it should "believe the typed exceptions of menger.failure.Fail" in:
    kindOf(EnvironmentException("ffmpeg not found")) shouldBe "environment"
    kindOf(InternalException("unreachable")) shouldBe "internal"
    kindOf(NotYetImplementedException("rebuilding a mixed scene")) shouldBe "scene"
    kindOf(UnknownGeometryException("blob", Seq("cube"))) shouldBe "scene"

  it should "call anything else a bug" in:
    kindOf(IllegalStateException("renderer not initialised")) shouldBe "internal"
    kindOf(NullPointerException()) shouldBe "internal"

  it should "keep the cause of an environment failure or a bug, for the stack trace" in:
    val gpu = gpuFailure("boom")
    val bug = IllegalStateException("unreachable")

    SceneFailure.of(gpu).cause shouldBe Some(gpu)
    SceneFailure.of(bug).cause shouldBe Some(bug)
    SceneFailure.of(IllegalArgumentException("bad")).cause shouldBe None

  it should "use the exception's type when it has no message" in:
    SceneFailure.of(NullPointerException()).message shouldBe "NullPointerException"

  // Main prints a stack trace for a bug only: a bad option or a missing ffmpeg needs its message.
  private def isBug(cause: Throwable): Boolean = SceneFailure.of(cause).isBug

  "SceneFailure.isBug" should "be true for a bug" in:
    isBug(IllegalStateException("renderer not initialised")) shouldBe true
    isBug(InternalException("unreachable mesh type")) shouldBe true

  it should "be false for the user's or the environment's fault" in:
    isBug(ConfigurationException("--display was given an empty value")) shouldBe false
    isBug(EnvironmentException("ffmpeg not found on PATH")) shouldBe false
    SceneFailure.Load(LoadError.NotFound("/x/scene.scala")).isBug shouldBe false

  "SceneFailure.Load" should "carry the load error's own text and no label" in:
    val failure = SceneFailure.Load(LoadError.NotFound("/x/scene.scala"))

    failure.kind shouldBe "load"
    failure.describe shouldBe "Scene file not found: /x/scene.scala"

  "SceneFailure.describe" should "label a failure that is not a load error with its kind" in:
    SceneFailure.of(IllegalStateException("oops")).describe shouldBe "[internal] oops"
    SceneFailure.of(IllegalArgumentException("bad size")).describe shouldBe "[scene] bad size"

  "SceneFailure.prefixed" should "put the step that failed in front of the message" in:
    SceneFailure.of(IllegalArgumentException("bad size")).prefixed("the new scene did not build")
      .describe shouldBe "[scene] the new scene did not build: bad size"
    SceneFailure.Load(LoadError.Malformed("no scene member")).prefixed("ignored").describe shouldBe
      "no scene member"
