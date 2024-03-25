package menger.objects

import menger.objects.Direction.Z
import org.scalatest.Assertions.assertThrows
import org.scalatest.flatspec.AnyFlatSpec

class SpongeBySurfaceSuite extends AnyFlatSpec:
  trait StartFace:
    val face: Face = Face(0, 0, 0, 1, Z)

  "surfaces at level 0" should "leave face intact" in new StartFace:
    val sponge: SpongeBySurface = SpongeBySurface(0)
    assert(sponge.surfaces(face).size == 1)
    assert(sponge.surfaces(face).head == face)

  "surfaces at level 1" should "create 12 subfaces" in new StartFace:
    val sponge: SpongeBySurface = SpongeBySurface(1)
    assert(sponge.surfaces(face).size == 12)

  "surfaces at level 2" should "create 12*12 subfaces" in new StartFace:
    val sponge: SpongeBySurface = SpongeBySurface(2)
    assert(sponge.surfaces(face).size == 12 * 12)
  
  "level -1" should "throw exception" in:
    assertThrows[IllegalArgumentException] {SpongeBySurface(-1)}
