package menger.objects

import menger.objects.Direction.X
import org.scalatest.funsuite.AnyFunSuite

class FaceSuite extends AnyFunSuite:
  test("instantiate Face") {
    val face = Face(0, 0, 0, 0, X)
    assert(face != null)
  }
