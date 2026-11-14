package examples.dsl.ladder

import menger.dsl.Vec3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class StageSuite extends AnyFlatSpec with Matchers:

  "The camera" should "stand on the sun's side of the object, 20 degrees round, 8 above" in:
    // The object is the star, not the sun: the sun sits behind the camera and lights the object's
    // face. The view runs along the sun's rays, 20 degrees to one side, tilted 8 degrees down.
    val rays = Look.default.keyLight.direction
    val horizontalRays = Vec3(rays.x, 0f, rays.z).normalize
    for t <- List(0f, 30f, 100f, 150f) do
      val camera = DimensionLadder.sceneAt(t).camera
      val view = (camera.lookAt - camera.position).normalize
      val horizontalView = Vec3(view.x, 0f, view.z).normalize
      val aside = math.toDegrees(math.acos(horizontalView.dot(horizontalRays).toDouble))
      withClue(s"t $t: "):
        aside shouldBe 20.0 +- 0.1
        math.toDegrees(math.asin(-view.y.toDouble)) shouldBe 8.0 +- 0.1
