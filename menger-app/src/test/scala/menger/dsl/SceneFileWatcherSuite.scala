package menger.dsl

import java.nio.file.Files
import java.util.concurrent.atomic.AtomicInteger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** F5 (usability review 2026-09): `InteractiveEngine`'s live scene reload needs a GPU context
  * to exercise end to end, so this suite covers the one GPU-free piece directly -- the file
  * watcher notices a change and calls back, coalescing a burst of writes into one callback.
  */
class SceneFileWatcherSuite extends AnyFlatSpec with Matchers:

  private val MaxWaitMillis = 5000L
  private val PollIntervalMillis = 50L

  @scala.annotation.tailrec
  private def awaitAtLeast(counter: AtomicInteger, min: Int, deadline: Long): Boolean =
    if counter.get() >= min then true
    else if System.currentTimeMillis() >= deadline then false
    else
      Thread.sleep(PollIntervalMillis)
      awaitAtLeast(counter, min, deadline)

  private def awaitAtLeast(counter: AtomicInteger, min: Int): Boolean =
    awaitAtLeast(counter, min, System.currentTimeMillis() + MaxWaitMillis)

  "SceneFileWatcher" should "invoke onChange after the watched file is modified" in:
    val dir = Files.createTempDirectory("scene-file-watcher-suite")
    val file = dir.resolve("scene.scala").toFile
    Files.writeString(file.toPath, "object Placeholder")
    val calls = new AtomicInteger(0)
    val watcher = new SceneFileWatcher(file, debounceMillis = 20L)(() => calls.incrementAndGet())
    try
      Files.writeString(file.toPath, "object Placeholder2")
      awaitAtLeast(calls, 1) shouldBe true
    finally watcher.close()

  it should "coalesce a burst of writes into a single callback" in:
    val dir = Files.createTempDirectory("scene-file-watcher-suite")
    val file = dir.resolve("scene.scala").toFile
    Files.writeString(file.toPath, "object Placeholder")
    val calls = new AtomicInteger(0)
    val watcher = new SceneFileWatcher(file, debounceMillis = 300L)(() => calls.incrementAndGet())
    try
      for i <- 1 to 5 do
        Files.writeString(file.toPath, s"object Placeholder$i")
        Thread.sleep(10L)
      // The debounce window (300ms) should still be running well after the last of these
      // rapid writes and before the callback fires -- give it time to fire exactly once.
      Thread.sleep(600L)
      calls.get() shouldBe 1
    finally watcher.close()

  it should "not invoke onChange for a different file in the same directory" in:
    val dir = Files.createTempDirectory("scene-file-watcher-suite")
    val watched = dir.resolve("scene.scala").toFile
    val other = dir.resolve("other.scala").toFile
    Files.writeString(watched.toPath, "object Placeholder")
    val calls = new AtomicInteger(0)
    val watcher = new SceneFileWatcher(watched, debounceMillis = 20L)(() => calls.incrementAndGet())
    try
      Files.writeString(other.toPath, "object Other")
      Thread.sleep(300L)
      calls.get() shouldBe 0
    finally watcher.close()

  it should "stop invoking onChange after close()" in:
    val dir = Files.createTempDirectory("scene-file-watcher-suite")
    val file = dir.resolve("scene.scala").toFile
    Files.writeString(file.toPath, "object Placeholder")
    val calls = new AtomicInteger(0)
    val watcher = new SceneFileWatcher(file, debounceMillis = 20L)(() => calls.incrementAndGet())
    watcher.close()
    Files.writeString(file.toPath, "object PlaceholderAfterClose")
    Thread.sleep(300L)
    calls.get() shouldBe 0
