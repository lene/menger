package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class WithPreviewSuite extends AnyFlatSpec with Matchers:

  "WithPreview.loopedT" should "follow the wall clock inside the time range" in:
    WithPreview.loopedT(elapsedSeconds = 2.5, startT = 0f, endT = 10f) shouldBe 2.5f

  it should "wrap around at the end of the range" in:
    WithPreview.loopedT(elapsedSeconds = 12.5, startT = 0f, endT = 10f) shouldBe 2.5f

  it should "offset by a non-zero start" in:
    WithPreview.loopedT(elapsedSeconds = 1.0, startT = 5f, endT = 7f) shouldBe 6f

  it should "stay at the start for an empty range" in:
    WithPreview.loopedT(elapsedSeconds = 3.0, startT = 4f, endT = 4f) shouldBe 4f

  // Usability review 2026-10, session 4 (F95): a slow frame made the real-time window skip the
  // rest of the loop; the every-frame playback renders every frame, and a window left open pauses.
  private val step = 1f / WithPreview.EveryFrameRate

  "WithPreview.nextEveryFrame" should "advance one step once the wall clock has caught up" in:
    WithPreview.nextEveryFrame(1f, step, elapsedSeconds = 1.05, startT = 0f, endT = 10f) shouldBe
      Some((1f + step, false))

  it should "wait while it is ahead of the wall clock (never faster than real time)" in:
    WithPreview.nextEveryFrame(1f, step, elapsedSeconds = 1.01, startT = 0f, endT = 10f) shouldBe None

  it should "not skip when frames are slow: one step per frame however late it is" in:
    // 8a: one frame took 2 s; the next t is still just one step on, not 2 s on.
    WithPreview.nextEveryFrame(10f, step, elapsedSeconds = 12.0, startT = 0f, endT = 12f) shouldBe
      Some((10f + step, false))

  it should "wrap to the start at the end of the loop" in:
    WithPreview.nextEveryFrame(11.99f, step, elapsedSeconds = 20.0, startT = 0f, endT = 12f) shouldBe
      Some((0f, true))

  it should "not move for a non-positive step" in:
    WithPreview.nextEveryFrame(1f, 0f, elapsedSeconds = 9.0, startT = 0f, endT = 10f) shouldBe None

  "WithPreview.skippedSeconds" should "report scene time jumped over by a slow frame" in:
    WithPreview.skippedSeconds(10.09f, 12.1f) shouldBe (2.01f +- 1e-4f)

  it should "report nothing for a normal step or a loop wrap" in:
    WithPreview.skippedSeconds(5f, 5.04f) shouldBe 0f
    WithPreview.skippedSeconds(11.9f, 0.05f) shouldBe 0f

  "WithPreview.isIdle" should "pause a window left alone past the limit, and not before" in:
    val second = 1000000000L
    WithPreview.isIdle(nowNanos = 601 * second, lastActivityNanos = 0L, idleSeconds = 600.0) shouldBe true
    WithPreview.isIdle(nowNanos = 599 * second, lastActivityNanos = 0L, idleSeconds = 600.0) shouldBe false
    WithPreview.isIdle(nowNanos = 601 * second, lastActivityNanos = 300 * second, idleSeconds = 600.0) shouldBe false

  it should "never pause when the limit is 0" in:
    WithPreview.isIdle(nowNanos = Long.MaxValue, lastActivityNanos = 0L, idleSeconds = 0.0) shouldBe false
