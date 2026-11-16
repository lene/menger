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
  private val step = 1.0 / WithPreview.EveryFrameRate

  "WithPreview.nextEveryFrame" should "advance one step once the wall clock has caught up" in:
    WithPreview.nextEveryFrame(1f, step, elapsedSeconds = 1.05, startT = 0f, endT = 10f) shouldBe
      Some((WithPreview.tAtFrame(26, 0.0, step), false))

  it should "wait while it is ahead of the wall clock (never faster than real time)" in:
    WithPreview.nextEveryFrame(1f, step, elapsedSeconds = 1.01, startT = 0f, endT = 10f) shouldBe None

  it should "not skip when frames are slow: one step per frame however late it is" in:
    // 8a: one frame took 2 s; the next t is still just one step on, not 2 s on.
    WithPreview.nextEveryFrame(10f, step, elapsedSeconds = 12.0, startT = 0f, endT = 12f) shouldBe
      Some((WithPreview.tAtFrame(251, 0.0, step), false))

  it should "wrap to the start at the end of the loop" in:
    WithPreview.nextEveryFrame(11.99, step, elapsedSeconds = 20.0, startT = 0f, endT = 12f) shouldBe
      Some((0f, true))

  it should "not move for a non-positive step" in:
    WithPreview.nextEveryFrame(1f, 0f, elapsedSeconds = 9.0, startT = 0f, endT = 10f) shouldBe None

  it should "step on the frame grid, not on the sum of the steps taken so far" in:
    // After 100000 frames summed in Float the time was off by seconds; a frame's time is its
    // index times the step, whatever was stepped before.
    val frame = 100000
    val t = WithPreview.tAtFrame(frame, 0.0, step)
    t shouldBe (frame * step +- 1e-9)
    WithPreview.nextEveryFrame(t, step, elapsedSeconds = 1e9, startT = 0.0, endT = 1e6) shouldBe
      Some((WithPreview.tAtFrame(frame + 1, 0.0, step), false))

  it should "snap a time between frames (after a key step) back onto the grid" in:
    WithPreview.nextEveryFrame(
      1.013, step, elapsedSeconds = 9.0, startT = 0.0, endT = 10.0
    ) shouldBe Some((WithPreview.tAtFrame(26, 0.0, step), false))

  "WithPreview.nextStepped" should "advance one frame and stop at the end" in:
    val frameStep = 0.1
    WithPreview.nextStepped(0.0, frameStep, 0.0, 1.0) shouldBe (0.1, false)
    WithPreview.nextStepped(0.8, frameStep, 0.0, 1.0)._1 shouldBe (0.9 +- 1e-12)
    WithPreview.nextStepped(0.9, frameStep, 0.0, 1.0) shouldBe (1.0, true)

  it should "reach the last frame exactly after as many steps as the range has frames" in:
    // 0..10 s in 101 frames: 100 steps of 0.1 summed in Float stop short of or overshoot 10.
    val (start, end, frameStep) = (0.0, 10.0, 0.1)
    val times = Iterator.iterate((start, false))((t, _) =>
      WithPreview.nextStepped(t, frameStep, start, end))
    val visited = times.take(200).takeWhile(!_._2).toList
    visited.size shouldBe 100
    visited.last._1 shouldBe (9.9 +- 1e-12)

  "WithPreview.frameIndexAt" should "name the frame a time is nearest to" in:
    WithPreview.frameIndexAt(0.26, 0.0, 0.1) shouldBe 3
    WithPreview.frameIndexAt(5.0, 5.0, 0.1) shouldBe 0
    WithPreview.frameIndexAt(3.0, 0.0, 0.0) shouldBe 0

  // The real-time playback measured its skipped time from a time of its own, which stepping and
  // jumping did not update: resuming after a jump counted the jump as skipped scene time.
  "WithPreview.realtimeTick" should "count no skipped time for a jump the playhead already made" in:
    val afterJump = 8.0
    val tick = WithPreview.realtimeTick(
      afterJump, 0.0, elapsedSeconds = 8.02, startT = 0.0, endT = 10.0)
    tick.t shouldBe 8.02
    tick.skippedTotal shouldBe 0.0

  it should "add the scene time a slow frame jumped over" in:
    val tick = WithPreview.realtimeTick(3.0, 1.0, elapsedSeconds = 5.0, startT = 0.0, endT = 10.0)
    tick.skippedTotal shouldBe 3.0

  it should "restart the total at a loop wrap" in:
    val tick = WithPreview.realtimeTick(9.9, 4.0, elapsedSeconds = 10.05, startT = 0.0, endT = 10.0)
    tick.t shouldBe (0.05 +- 1e-9)
    tick.skippedTotal shouldBe 0.0

  // A key step or a jump while playing moves t; the clock must move with it, or the next tick
  // throws t back to the clock's time and counts the distance as skipped scene time.
  "WithPreview.clockStartNanos" should "continue the real-time playback from a t moved while playing" in:
    val second = 1000000000L
    val now = 100 * second
    def tickAfterMove(movedTo: Double, laterSeconds: Double) =
      val start = WithPreview.clockStartNanos(now, movedTo, startT = 0.0)
      val elapsed = (now + (laterSeconds * second).toLong - start) / WithPreview.NanosPerSecond
      WithPreview.realtimeTick(movedTo, 0.0, elapsed, startT = 0.0, endT = 10.0)

    // Stepped back half a second from t = 6, and jumped to the start.
    val stepped = tickAfterMove(5.5, laterSeconds = 0.02)
    stepped.t shouldBe (5.52 +- 1e-6)
    stepped.skippedTotal shouldBe 0.0
    val jumped = tickAfterMove(0.0, laterSeconds = 0.02)
    jumped.t shouldBe (0.02 +- 1e-6)
    jumped.skippedTotal shouldBe 0.0

  it should "start the clock as long ago as the playback has run from the start of the range" in:
    val second = 1000000000L
    WithPreview.clockStartNanos(10 * second, 4.0, startT = 1.0) shouldBe 7 * second

  "WithPreview.skippedSeconds" should "report scene time jumped over by a slow frame" in:
    WithPreview.skippedSeconds(10.09, 12.1) shouldBe (2.01 +- 1e-9)

  it should "report nothing for a normal step or a loop wrap" in:
    WithPreview.skippedSeconds(5f, 5.04f) shouldBe 0.0
    WithPreview.skippedSeconds(11.9, 0.05) shouldBe 0.0

  "WithPreview.isIdle" should "pause a window left alone past the limit, and not before" in:
    val second = 1000000000L
    WithPreview.isIdle(nowNanos = 601 * second, lastActivityNanos = 0L, idleSeconds = 600.0) shouldBe true
    WithPreview.isIdle(nowNanos = 599 * second, lastActivityNanos = 0L, idleSeconds = 600.0) shouldBe false
    WithPreview.isIdle(nowNanos = 601 * second, lastActivityNanos = 300 * second, idleSeconds = 600.0) shouldBe false

  it should "never pause when the limit is 0" in:
    WithPreview.isIdle(nowNanos = Long.MaxValue, lastActivityNanos = 0L, idleSeconds = 0.0) shouldBe false

  // Usability review 2026-10, session 5 (F125, menger#133): 3.4 s frames with nothing said.
  "WithPreview.isSlow" should "report a frame over a second, at most every five seconds" in:
    val second = 1000000000L
    WithPreview.isSlow(frameNanos = 2 * second, lastReportNanos = None, now = 0L) shouldBe true
    WithPreview.isSlow(frameNanos = second / 2, lastReportNanos = None, now = 0L) shouldBe false
    WithPreview.isSlow(2 * second, lastReportNanos = Some(10 * second), now = 13 * second) shouldBe false
    WithPreview.isSlow(2 * second, lastReportNanos = Some(10 * second), now = 15 * second) shouldBe true

  // Session 5, F125 (menger#133): a slowly rendering window looked blank or frozen.
  "WithPreview.slowFrameNote" should "name a slow frame's time in whole seconds, else nothing" in:
    val second = 1000000000L
    WithPreview.slowFrameNote(11 * second + second / 4) shouldBe Some("slow: last frame 11 s")
    WithPreview.slowFrameNote(second / 2) shouldBe None

  "WithPreview.renderingFrameNote" should "name the frame being rendered" in:
    WithPreview.renderingFrameNote(0, 251) shouldBe "rendering frame 0/251…"

  "HeapUsage.describe" should "name the used and the maximum heap in GiB" in:
    HeapUsage.describe() should fullyMatch regex """heap \d+\.\d of \d+\.\d GiB"""
