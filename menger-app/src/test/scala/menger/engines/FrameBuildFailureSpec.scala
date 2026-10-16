package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Usability review 2026-09, session 2 (menger#54): a frame whose scene failed to build was
  * only logged in free text, so the scene agent never learned that the window kept showing the
  * previous frame. The log line now carries a fixed marker the agent can watch for. */
class FrameBuildFailureSpec extends AnyFlatSpec with Matchers:

  "FrameBuildFailure.message" should "start with the fixed marker and name the frame and cause" in:
    val line = FrameBuildFailure.message("t=1.5", IllegalStateException("level out of range"))
    line should startWith("FRAME-BUILD-FAILED ")
    line should include("t=1.5")
    line should include("level out of range")

  it should "fall back to the exception's type when it has no message" in:
    FrameBuildFailure.message("frame 3", NullPointerException()) should
      endWith("frame 3: NullPointerException")
