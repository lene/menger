package menger.engines.scene

import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicInteger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** The memo that bounds the unprojected n-sponge meshes by the faces they retain (menger#166). */
class FaceBoundedMemoSuite extends AnyFlatSpec with Matchers:

  private val limit = AtomicLong(10L)
  private val builds = AtomicInteger(0)

  private def memo = FaceBoundedMemo[Int, String](() => limit.get, _.length.toLong)

  private def get(m: FaceBoundedMemo[Int, String], key: Int, weight: Int): String =
    m.getOrElseUpdate(key) {
      val _ = builds.incrementAndGet()
      "x" * weight
    }

  "A FaceBoundedMemo" should "build a value once and return the same value again" in:
    limit.set(10L)
    val m = memo
    val first = get(m, 1, 3)
    get(m, 1, 3) should be theSameInstanceAs first
    m.size shouldBe 1
    m.retainedWeight shouldBe 3L

  it should "evict the oldest entries until the retained weight is within the limit" in:
    limit.set(10L)
    val m = memo
    val first = get(m, 1, 4)
    get(m, 2, 4)
    m.retainedWeight shouldBe 8L
    get(m, 3, 4)
    m.size shouldBe 2
    m.retainedWeight shouldBe 8L
    get(m, 1, 4) should not be theSameInstanceAs(first)

  it should "evict the least recently used entry, not the first one inserted" in:
    limit.set(10L)
    val m = memo
    val first = get(m, 1, 4)
    get(m, 2, 4)
    get(m, 1, 4) should be theSameInstanceAs first
    get(m, 3, 4)
    get(m, 1, 4) should be theSameInstanceAs first
    m.size shouldBe 2

  it should "keep the newest entry even when it alone is over the limit" in:
    limit.set(10L)
    val m = memo
    get(m, 1, 4)
    val big = get(m, 2, 50)
    m.size shouldBe 1
    get(m, 2, 50) should be theSameInstanceAs big

  it should "read the limit on every insert" in:
    limit.set(100L)
    val m = memo
    (1 to 5).foreach(get(m, _, 10))
    m.size shouldBe 5
    limit.set(25L)
    get(m, 6, 10)
    m.retainedWeight should be <= 25L
    m.size shouldBe 2

  it should "build nothing again for entries it keeps" in:
    limit.set(100L)
    val m = memo
    builds.set(0)
    (1 to 5).foreach(get(m, _, 10))
    (1 to 5).foreach(get(m, _, 10))
    builds.get shouldBe 5
