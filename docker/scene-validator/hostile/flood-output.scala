// An otherwise valid scene that prints 10 MB on standard output (menger#137), most of it in one
// line without a line break: the validator caps what it keeps of it and still returns the real
// verdict (`ok`).
import menger.dsl._
object FloodOutput:
  private val megabyte = "x" * (1024 * 1024)
  (1 to 5).foreach(_ => print(megabyte))
  println()
  (1 to 5).foreach(_ => println(megabyte))
  val scene: Scene = Scene(
    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
    objects = List(Sphere()), lights = List(Directional((1f, -1f, -1f))))
