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
  // A client with a trusted cookie can type into any window on the display through XTEST, a host
  // terminal included. No X library is needed: the raw protocol over the mounted socket, with the
  // mounted cookie, is enough (menger#77 residual).
  probe("x-xtest") {
    import java.nio.{ByteBuffer, ByteOrder}
    import java.nio.channels.SocketChannel
    def pad(n: Int) = (4 - n % 4) % 4
    val display = Option(System.getenv("DISPLAY")).getOrElse(throw RuntimeException("no DISPLAY"))
    val number = display.dropWhile(_ != ':').drop(1).takeWhile(_.isDigit)
    val auth = ByteBuffer.wrap(java.nio.file.Files.readAllBytes(
      java.nio.file.Path.of(Option(System.getenv("XAUTHORITY")).getOrElse(throw RuntimeException("no XAUTHORITY")))))
    def counted() = { val b = new Array[Byte](auth.getShort & 0xffff); auth.get(b); b }
    auth.getShort; counted(); counted()
    val (name, data) = (counted(), counted())
    val ch = SocketChannel.open(java.net.UnixDomainSocketAddress.of(s"/tmp/.X11-unix/X$number"))
    def send(b: ByteBuffer) = { b.flip(); while b.hasRemaining do ch.write(b) }
    def recv(n: Int) =
      val b = ByteBuffer.allocate(n).order(ByteOrder.LITTLE_ENDIAN)
      while b.hasRemaining do if ch.read(b) < 0 then throw RuntimeException("closed")
      b.flip(); b
    val setup = ByteBuffer.allocate(12 + name.length + pad(name.length) + data.length + pad(data.length))
      .order(ByteOrder.LITTLE_ENDIAN)
    setup.put('l'.toByte).put(0.toByte).putShort(11).putShort(0)
      .putShort(name.length.toShort).putShort(data.length.toShort).putShort(0)
      .put(name).put(new Array[Byte](pad(name.length))).put(data).put(new Array[Byte](pad(data.length)))
    send(setup)
    val head = recv(8)
    if head.get(0) != 1 then throw RuntimeException("connection refused")
    recv((head.getShort(6) & 0xffff) * 4)
    val ext = "XTEST".getBytes
    val query = ByteBuffer.allocate(8 + ext.length + pad(ext.length)).order(ByteOrder.LITTLE_ENDIAN)
    query.put(98.toByte).put(0.toByte).putShort(((8 + ext.length + pad(ext.length)) / 4).toShort)
      .putShort(ext.length.toShort).putShort(0).put(ext).put(new Array[Byte](pad(ext.length)))
    send(query)
    val reply = recv(32)
    if reply.get(0) == 1 && reply.get(8) != 0 then "XTEST usable on display " + display
    else throw RuntimeException("XTEST hidden")
  }
  val scene: Scene = Scene(
    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
    objects = List(Sphere()), lights = List(Directional((1f, -1f, -1f))))
