package menger.failure

import menger.common.ConfigurationException
import menger.common.EnvironmentException
import menger.common.InternalException
import menger.common.NotYetImplementedException
import menger.common.ParseException
import menger.common.UnknownGeometryException
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** menger#96, #103: `sys.error` raised a bare RuntimeException for three different causes. Each
  * Fail method says whose fault it is by the exception it throws. */
class FailSpec extends AnyFlatSpec with Matchers:

  "Fail.configuration" should "throw a ConfigurationException with the message" in:
    val e = intercept[ConfigurationException](Fail.configuration("--display was given an empty value"))

    e.getMessage shouldBe "--display was given an empty value"

  "Fail.parse" should "throw a ParseException naming the input and what was expected" in:
    val e = intercept[ParseException](Fail.parse("Unknown tag", "bogus", "ok or refused"))

    e.getMessage shouldBe "Unknown tag. Input: 'bogus'. Expected: ok or refused"

  "Fail.unknownGeometry" should "throw an UnknownGeometryException listing the valid types" in:
    val e = intercept[UnknownGeometryException](Fail.unknownGeometry("blob", Seq("cube", "sphere")))

    e.getMessage shouldBe "Unknown geometry type: 'blob'. Valid types: cube, sphere"

  "Fail.unsupported" should "throw a NotYetImplementedException naming the feature" in:
    val e = intercept[NotYetImplementedException](Fail.unsupported("rebuilding a mixed scene"))

    e.getMessage shouldBe "Not yet implemented: rebuilding a mixed scene"

  "Fail.environment" should "carry the remedy after the problem" in:
    val e = intercept[EnvironmentException](
      Fail.environment("ffmpeg not found on PATH", "Install ffmpeg to use --video output")
    )

    e.getMessage shouldBe "ffmpeg not found on PATH. Install ffmpeg to use --video output"

  it should "keep the cause" in:
    val cause = IllegalStateException("context lost")

    val e = intercept[EnvironmentException](
      Fail.environment("CUDA error", "Restart the application", Some(cause))
    )

    e.getCause shouldBe cause

  it should "need no remedy when the user can do nothing" in:
    intercept[EnvironmentException](Fail.environment("native library failed to load"))
      .getMessage shouldBe "native library failed to load"

  "Fail.internal" should "throw an InternalException that says it is a bug" in:
    intercept[InternalException](Fail.internal("unreachable mesh type")).getMessage shouldBe
      "Internal error: unreachable mesh type"
