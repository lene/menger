package menger.objects

import com.badlogic.gdx.math.Vector3
import menger.ProfilingConfig
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LogTimeSuite extends AnyFlatSpec with Matchers:

  // Test geometry that uses logTime
  class TestGeometry()(using config: ProfilingConfig) extends Geometry(Vector3.Zero, 1f):
    // Use AtomicInteger for thread-safe counting without var
    private val counter = java.util.concurrent.atomic.AtomicInteger(0)

    override def getModel = Nil

    def timedOperation(delayMs: Int): String =
      logTime("test-operation") {
        val count = counter.incrementAndGet()
        Thread.sleep(delayMs)
        s"result-$count"
      }

    def executionCount: Int = counter.get()

  "logTime with disabled profiling" should "execute the block" in:
    given ProfilingConfig = ProfilingConfig.disabled
    val geom = TestGeometry()

    val result = geom.timedOperation(0)

    result shouldBe "result-1"
    geom.executionCount shouldBe 1

  it should "return the block's value" in:
    given ProfilingConfig = ProfilingConfig.disabled
    val geom = TestGeometry()

    val result1 = geom.timedOperation(0)
    val result2 = geom.timedOperation(0)

    result1 shouldBe "result-1"
    result2 shouldBe "result-2"

  "logTime with enabled profiling" should "execute the block" in:
    given ProfilingConfig = ProfilingConfig.enabled(10)
    val geom = TestGeometry()

    val result = geom.timedOperation(0)

    result shouldBe "result-1"
    geom.executionCount shouldBe 1

  it should "return the block's value" in:
    given ProfilingConfig = ProfilingConfig.enabled(10)
    val geom = TestGeometry()

    val result = geom.timedOperation(0)

    result shouldBe "result-1"

  it should "work with operations taking longer than threshold" in:
    given ProfilingConfig = ProfilingConfig.enabled(10)
    val geom = TestGeometry()

    // Operation takes ~50ms, threshold is 10ms
    // Should execute and return correctly even though it exceeds threshold
    val result = geom.timedOperation(50)

    result shouldBe "result-1"
    geom.executionCount shouldBe 1

  "logTime with threshold 0" should "execute the block" in:
    given ProfilingConfig = ProfilingConfig.enabled(0)
    val geom = TestGeometry()

    val result = geom.timedOperation(0)

    result shouldBe "result-1"
    geom.executionCount shouldBe 1

  it should "log all operations regardless of duration" in:
    given ProfilingConfig = ProfilingConfig.enabled(0)
    val geom = TestGeometry()

    // With threshold 0, even instant operations (0ms) should be logged
    // We can't easily verify log output in tests, but we verify the operation executes
    val result1 = geom.timedOperation(0)
    val result2 = geom.timedOperation(1)

    result1 shouldBe "result-1"
    result2 shouldBe "result-2"
    geom.executionCount shouldBe 2

  "logTime" should "not affect the result regardless of config" in:
    val results = for {
      config <- List(ProfilingConfig.disabled, ProfilingConfig.enabled(0), ProfilingConfig.enabled(1000))
    } yield {
      given ProfilingConfig = config
      val geom = TestGeometry()
      geom.timedOperation(0)
    }

    // All configs should produce the same result
    results.toSet shouldBe Set("result-1")

  it should "allow nested calls" in:
    given ProfilingConfig = ProfilingConfig.disabled

    class NestedGeometry()(using config: ProfilingConfig) extends Geometry(Vector3.Zero, 1f):
      override def getModel = Nil

      def outer(): Int =
        logTime("outer") {
          inner() + 10
        }

      def inner(): Int =
        logTime("inner") {
          42
        }

    val geom = NestedGeometry()
    geom.outer() shouldBe 52
