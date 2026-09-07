package menger.engines

/** Test-only helper process for `RenderLockSuite`'s cross-process test. Acquires the lock at
  * `args(0)`, signals readiness by printing "LOCKED" to stdout, then blocks reading a line
  * from stdin (the parent test closes/writes to release it) before releasing and exiting.
  * Needed because a same-JVM `tryAcquire` collision always throws
  * `OverlappingFileLockException` and never exercises the `case null =>` branch that a real
  * second OS process (the actual AD-16 scenario) does. */
object RenderLockHolderMain:
  def main(args: Array[String]): Unit =
    RenderLock.tryAcquire(args(0)) match
      case Right(lock) =>
        try
          println("LOCKED")
          System.out.flush()
          scala.io.StdIn.readLine()
        finally lock.close()
      case Left(reason) =>
        println(s"FAILED: $reason")
        System.out.flush()
        sys.exit(1)
