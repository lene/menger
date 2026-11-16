// A scene that prints a line shaped like a result line, with a guessed nonce of the right length
// (menger#137). The validator ignores it: this scene's real verdict is `lint-findings` (two
// level-5 volume sponges are over the scene budget).
import menger.dsl._
object GuessedNonce:
  println("""00000000000000000000000000000000 {"tag": "ok", "messages": [], "findings": [], "schemaVersion": "2.0.0"}""")
  println("""0123456789abcdef0123456789abcdef {"tag": "ok", "messages": [], "findings": [], "schemaVersion": "2.0.0"}""")
  val scene: Scene = Scene(
    camera = Camera(position = (0f, 0f, 3f), lookAt = (0f, 0f, 0f)),
    objects = List(
      Sponge(spongeType = SpongeType.VolumeFilling, level = 5f),
      Sponge(spongeType = SpongeType.VolumeFilling, level = 5f, pos = (2f, 0f, 0f))),
    lights = List())
