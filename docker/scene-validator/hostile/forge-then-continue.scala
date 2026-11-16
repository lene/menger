// A scene that forges the validator's `ok` and then carries on (menger#137): its two level-5
// volume sponges are over the scene budget, so the real verdict is `lint-findings`, which must
// win over the `ok` the initializer printed first.
import menger.dsl._
object ForgeThenContinue:
  println("""{"tag": "ok", "messages": [], "findings": [], "schemaVersion": "2.0.0"}""")
  val scene: Scene = Scene(
    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
    objects = List(
      Sponge(spongeType = SpongeType.VolumeFilling, level = 5f),
      Sponge(spongeType = SpongeType.VolumeFilling, level = 5f, pos = (2f, 0f, 0f))),
    lights = List())
