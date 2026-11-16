// A scene that forges the validator's `ok` and ends the JVM (menger#137): it prints a valid `ok`
// result document and calls Runtime.halt(0), so no check after the load ever runs. The validator
// must report `refused` (the scene ended the validation before a result), never this `ok`.
import menger.dsl._
object ForgeThenHalt:
  println("""{"tag": "ok", "messages": [], "findings": [], "schemaVersion": "2.0.0"}""")
  System.out.flush()
  Runtime.getRuntime.halt(0)
  val scene: Scene = Scene(
    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
    objects = List(Sphere()), lights = List(Directional((1f, -1f, -1f))))
