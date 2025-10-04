package menger.objects

import com.badlogic.gdx.math.Vector3
import menger.objects.Direction.Z
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SpongeBySurfaceSuite extends AnyFlatSpec with Matchers:
  trait StartFace:
    val face: Face = Face(0, 0, 0, 1, Z)

  "surfaces at level 0" should "leave face intact" in new StartFace:
    val sponge: SpongeBySurface = SpongeBySurface(Vector3.Zero, 1f, 0)
    sponge.surfaces(face) should have size 1
    sponge.surfaces(face).head should be (face)

  "surfaces at level 1" should "create 12 subfaces" in new StartFace:
    val sponge: SpongeBySurface = SpongeBySurface(Vector3.Zero, 1f, 1)
    sponge.surfaces(face) should have size 12

  "surfaces at level 2" should "create 12*12 subfaces" in new StartFace:
    val sponge: SpongeBySurface = SpongeBySurface(Vector3.Zero, 1f, 2)
    sponge.surfaces(face) should have size 12 * 12
  
  "level -1" should "throw exception" in:
    an[IllegalArgumentException] should be thrownBy SpongeBySurface(Vector3.Zero, 1f, -1)

  "fractional level 1.5" should "instantiate" in new StartFace:
    val sponge: SpongeBySurface = SpongeBySurface(Vector3.Zero, 1f, 1.5f)
    sponge.level shouldBe 1.5f

  it should "have a transparentSponge" in new StartFace:
    val sponge: SpongeBySurface = SpongeBySurface(Vector3.Zero, 1f, 1.5f)
    sponge.transparentSponge should not be empty
    sponge.transparentSponge.get.level shouldBe 1.0f

  it should "have a nextLevelSponge" in new StartFace:
    val sponge: SpongeBySurface = SpongeBySurface(Vector3.Zero, 1f, 1.5f)
    sponge.nextLevelSponge should not be empty
    sponge.nextLevelSponge.get.level shouldBe 2.0f
