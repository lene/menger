package menger.objects

import menger.objects.Direction.{X, negX, Y, negY, Z, negZ}
import org.scalatest.funsuite.AnyFunSuite

class DirectionSuite extends AnyFunSuite:

  test("instantiate from valid values") {
    assert(Direction(1, 0, 0) == X)
    assert(Direction(0, 1, 0) == Y)
    assert(Direction(0, 0, 1) == Z)
    assert(Direction(-1, 0, 0) == negX)
    assert(Direction(0, -1, 0) == negY)
    assert(Direction(0, 0, -1) == negZ)
  }

  test("instantiate from valid values as tuple") {
    assert(Direction((1, 0, 0)) == X)
    assert(Direction((0, 1, 0)) == Y)
    assert(Direction((0, 0, 1)) == Z)
    assert(Direction((-1, 0, 0)) == negX)
    assert(Direction((0, -1, 0)) == negY)
    assert(Direction((0, 0, -1)) == negZ)
  }

  test("instantiate from invalid values throws IllegalArgumentException") {
    assertThrows[IllegalArgumentException](Direction(0, 0, 0))
    assertThrows[IllegalArgumentException](Direction(1, 1, 0))
    assertThrows[IllegalArgumentException](Direction(2, 0, 0))
    assertThrows[IllegalArgumentException](Direction(-1, -1, 0))
    assertThrows[IllegalArgumentException](Direction(-2, 0, 0))
  }

  test("negative Direction") {
    assert(-X == negX)
    assert(-Y == negY)
    assert(-Z == negZ)
    assert(-negX == X)
    assert(-negY == Y)
    assert(-negZ == Z)
  }

  test("accessing components of a Direction") {
    assert(X.x == 1)
    assert(X.y == 0)
    assert(X.z == 0)
    assert(Y.x == 0)
    assert(Y.y == 1)
    assert(Y.z == 0)
    assert(Z.x == 0)
    assert(Z.y == 0)
    assert(Z.z == 1)
  }

  test("rotation around X axis") {
    assert(X.rotate90(X) == X)
    assert(Y.rotate90(X) == Z)
    assert(Z.rotate90(X) == negY)
    assert(negX.rotate90(X) == negX)
    assert(negY.rotate90(X) == negZ)
    assert(negZ.rotate90(X) == Y)
  }

  test("rotation around Y axis") {
    assert(X.rotate90(Y) == negZ)
    assert(Y.rotate90(Y) == Y)
    assert(Z.rotate90(Y) == X)
    assert(negX.rotate90(Y) == Z)
    assert(negY.rotate90(Y) == negY)
    assert(negZ.rotate90(Y) == negX)
  }

  test("rotation around Z axis") {
    assert(X.rotate90(Z) == Y)
    assert(Y.rotate90(Z) == negX)
    assert(Z.rotate90(Z) == Z)
    assert(negX.rotate90(Z) == negY)
    assert(negY.rotate90(Z) == X)
    assert(negZ.rotate90(Z) == negZ)
  }

  test("rotating around -X axis") {
    assert(X.rotate90(negX) == X)
    assert(Y.rotate90(negX) == negZ)
    assert(Z.rotate90(negX) == Y)
    assert(negX.rotate90(negX) == negX)
    assert(negY.rotate90(negX) == Z)
    assert(negZ.rotate90(negX) == negY)
  }

  test("rotating around -Y axis") {
    assert(X.rotate90(negY) == Z)
    assert(Y.rotate90(negY) == Y)
    assert(Z.rotate90(negY) == negX)
    assert(negX.rotate90(negY) == negZ)
    assert(negY.rotate90(negY) == negY)
    assert(negZ.rotate90(negY) == X)
  }

  test("rotating around -Z axis") {
    assert(X.rotate90(negZ) == negY)
    assert(Y.rotate90(negZ) == X)
    assert(Z.rotate90(negZ) == Z)
    assert(negX.rotate90(negZ) == Y)
    assert(negY.rotate90(negZ) == negX)
    assert(negZ.rotate90(negZ) == negZ)
  }

  test("signs of all constants") {
    assert(X.sign == 1)
    assert(Y.sign == 1)
    assert(Z.sign == 1)
    assert(negX.sign == -1)
    assert(negY.sign == -1)
    assert(negZ.sign == -1)
  }

  test("absolute values of all constants") {
    assert(X.abs == X)
    assert(Y.abs == Y)
    assert(Z.abs == Z)
    assert(negX.abs == X)
    assert(negY.abs == Y)
    assert(negZ.abs == Z)
  }

  test("fold1") {
    assert(X.fold1 == -Y)
    assert(Y.fold1 == -Z)
    assert(Z.fold1 == -X)
  }

  test("fold2") {
    assert(X.fold2 == -Z)
    assert(Y.fold2 == -X)
    assert(Z.fold2 == -Y)
  }