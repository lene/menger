package menger.engines

import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.file.Files
import java.nio.file.Paths

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RenderLockSuite extends AnyFlatSpec with Matchers:

  private def freshLockPath(): String =
    val dir = Files.createTempDirectory("render-lock-suite")
    dir.resolve("render.lock").toString

  "RenderLock.tryAcquire" should "succeed on a fresh path" in:
    val path = freshLockPath()
    val result = RenderLock.tryAcquire(path)
    result shouldBe a[Right[?, ?]]
    result.foreach(_.close())

  it should "fail cleanly (not throw) when the path is already locked" in:
    val path = freshLockPath()
    val first = RenderLock.tryAcquire(path)
    first shouldBe a[Right[?, ?]]
    try
      val second = RenderLock.tryAcquire(path)
      second shouldBe a[Left[?, ?]]
      second.left.foreach(_ should include(path))
    finally
      first.foreach(_.close())

  it should "free the path for a subsequent acquire once released" in:
    val path = freshLockPath()
    val first = RenderLock.tryAcquire(path)
    first shouldBe a[Right[?, ?]]
    first.foreach(_.close())

    val second = RenderLock.tryAcquire(path)
    second shouldBe a[Right[?, ?]]
    second.foreach(_.close())

  it should "report a clear failure naming the path when its directory does not exist" in:
    val dir = Files.createTempDirectory("render-lock-suite-missing")
    val missing = dir.resolve("no-such-subdir").resolve("render.lock").toString
    val result = RenderLock.tryAcquire(missing)
    result shouldBe a[Left[?, ?]]
    result.left.foreach(_ should include(missing))

  it should "allow close() to be called more than once without throwing" in:
    val path = freshLockPath()
    val result = RenderLock.tryAcquire(path)
    result shouldBe a[Right[?, ?]]
    result.foreach { lock =>
      lock.close()
      noException should be thrownBy lock.close()
    }

  // The two tests above only ever collide within this same JVM, where the JDK always throws
  // `OverlappingFileLockException` -- they never exercise `tryLock() == null`, the actual
  // cross-process AD-16 scenario, which is the only genuinely novel runtime behavior this
  // story adds (review round). This test spawns a real second JVM to hold the lock instead.
  "RenderLock.tryAcquire across processes" should "fail cleanly when a different process holds the lock" in:
    val path = freshLockPath()
    val javaBin = Paths.get(System.getProperty("java.home"), "bin", "java").toString
    val classpath = System.getProperty("java.class.path")
    val holder = ProcessBuilder(javaBin, "-cp", classpath, "menger.engines.RenderLockHolderMain", path)
      .redirectErrorStream(true)
      .start()
    try
      val out = BufferedReader(InputStreamReader(holder.getInputStream))
      val readiness = out.readLine()
      readiness shouldEqual "LOCKED"

      val result = RenderLock.tryAcquire(path)
      result shouldBe a[Left[?, ?]]
      result.left.foreach(_ should include(path))
    finally
      holder.getOutputStream.close() // unblocks the holder's readLine(), letting it release+exit
      holder.waitFor()
