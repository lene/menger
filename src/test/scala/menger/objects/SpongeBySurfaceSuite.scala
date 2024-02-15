package menger.objects

import menger.objects.Direction.Z
import org.scalatest.funsuite.AnyFunSuite

class SpongeBySurfaceSuite extends AnyFunSuite:

  test("surfaces at level 0 leaves face intact") {
    val sponge = SpongeBySurface(0)
    val face = Face(0, 0, 0, 1, Z)
    assert(sponge.surfaces(face).size == 1)
    assert(sponge.surfaces(face).head == face)
  }

  test("surfaces at level 1 creates 12 subfaces") {
    val sponge = SpongeBySurface(1)
    val face = Face(0, 0, 0, 1, Z)
    assert(sponge.surfaces(face).size == 12)
  }

  test("surfaces at level 2 creates 12*12 subfaces") {
    val sponge = SpongeBySurface(2)
    val face = Face(0, 0, 0, 1, Z)
    assert(sponge.surfaces(face).size == 12 * 12)
  }
  
