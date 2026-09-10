package menger.engines

/** Test-only helper process for `RenderLockSuite`'s cross-process test. Acquires the lock at
  * `args(0)`, signals readiness by printing "LOCKED" to stdout, then blocks reading a line
  * from stdin (the parent test closes/writes to release it) before releasing and exiting.
  * Needed because a same-JVM `tryAcquire` collision always throws
  * `OverlappingFileLockException` and never exercises the `case null =>` branch that a real
  * second OS process (the actual AD-16 scenario) does. */
object RenderLockHolderMain:
  def main(args: Array[String]): Unit =
    // Review round 2: `args(0)` unguarded raised ArrayIndexOutOfBoundsException instead of the
    // FAILED line the parent test reads for, so a mis-invoked helper hung the parent on
    // readLine() rather than failing it.
    args.headOption match
      case None =>
        println("FAILED: no lock path given")
        System.out.flush()
        sys.exit(2)
      case Some(path) => acquireAndHold(path)

  private def acquireAndHold(path: String): Unit =
    RenderLock.tryAcquire(path) match
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
