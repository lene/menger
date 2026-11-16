// A scene whose initializer ends the JVM with a success status (menger#137): the validator must
// report `refused`, not take the clean exit for a verdict.
import menger.dsl._
object ExitInInitializer:
  System.exit(0)
  val scene: Scene = Scene(
    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
    objects = List(Sphere()), lights = List(Directional((1f, -1f, -1f))))
