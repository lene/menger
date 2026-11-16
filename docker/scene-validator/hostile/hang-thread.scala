// An otherwise valid scene that starts a non-daemon thread that never ends (menger#137): the JVM
// would stay alive until the outer time limit. The verdict (`ok`) must arrive without waiting
// for it.
import menger.dsl._
object HangThread:
  private val forever = new Thread(() => Thread.sleep(Long.MaxValue))
  forever.setDaemon(false)
  forever.start()
  val scene: Scene = Scene(
    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
    objects = List(Sphere()), lights = List(Directional((1f, -1f, -1f))))
