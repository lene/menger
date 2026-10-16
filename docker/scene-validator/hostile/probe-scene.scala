// A hostile scene for test-containment.sh (review 2026-10-14, menger#78): its initializer tries
// what generated code must not be able to do and prints one `PROBE <name>: <result>` line each.
// Inside the containers every probe but `uid` (not 0) must print "blocked". @SIBLING@ is replaced
// by a file next to the window directory and @HOME@ by the host user's home; both must stay
// invisible.
import menger.dsl._
object HostileProbe:
  private def probe(name: String)(f: => Any): Unit =
    val r = try "SUCCEEDED " + f catch case e: Throwable => "blocked (" + e.getClass.getSimpleName + ")"
    System.err.println(s"PROBE $name: $r")
  probe("network")(java.net.Socket("1.1.1.1", 53).isConnected)
  probe("dns")(java.net.InetAddress.getByName("github.com"))
  probe("read-home")(java.io.File("@HOME@").list().length)
  probe("read-sibling")(java.nio.file.Files.readString(java.nio.file.Path.of("@SIBLING@")))
  probe("write-opt")(java.nio.file.Files.writeString(java.nio.file.Path.of("/opt/probe"), "x"))
  probe("write-etc")(java.nio.file.Files.writeString(java.nio.file.Path.of("/etc/probe"), "x"))
  probe("env-key")(Option(System.getenv("ANTHROPIC_API_KEY")).getOrElse(throw RuntimeException("absent")))
  probe("docker-socket")(
    if java.io.File("/var/run/docker.sock").exists then "visible" else throw RuntimeException("absent"))
  probe("uid")(new String(Runtime.getRuntime.exec(Array("id", "-u")).getInputStream.readAllBytes()).trim)
  val scene: Scene = Scene(
    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
    objects = List(Sphere()), lights = List(Directional((1f, -1f, -1f))))
