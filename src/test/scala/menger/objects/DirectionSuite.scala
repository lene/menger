package menger.objects

import menger.objects.Direction.{X, Y, Z, negX, negY, negZ}
import org.scalatest.flatspec.AnyFlatSpec

class DirectionSuite extends AnyFlatSpec:

  "instantiating from valid values" should "work" in:
    assert(Direction(1, 0, 0) == X)
    assert(Direction(0, 1, 0) == Y)
    assert(Direction(0, 0, 1) == Z)
    assert(Direction(-1, 0, 0) == negX)
    assert(Direction(0, -1, 0) == negY)
    assert(Direction(0, 0, -1) == negZ)

  it should "work from tuple" in:
    assert(Direction((1, 0, 0)) == X)
    assert(Direction((0, 1, 0)) == Y)
    assert(Direction((0, 0, 1)) == Z)
    assert(Direction((-1, 0, 0)) == negX)
    assert(Direction((0, -1, 0)) == negY)
    assert(Direction((0, 0, -1)) == negZ)

  "instantiating from invalid values" should "throw IllegalArgumentException for zero vector" in:
    assertThrows[IllegalArgumentException](Direction(0, 0, 0))
  it should "throw IllegalArgumentException for non-unit vectors" in:
    assertThrows[IllegalArgumentException](Direction(1, 1, 0))
    assertThrows[IllegalArgumentException](Direction(2, 0, 0))
    assertThrows[IllegalArgumentException](Direction(-1, -1, 0))
    assertThrows[IllegalArgumentException](Direction(-2, 0, 0))

  "minus on Direction" should "return its negative" in:
    assert(-X == negX)
    assert(-Y == negY)
    assert(-Z == negZ)
    assert(-negX == X)
    assert(-negY == Y)
    assert(-negZ == Z)

  "accessing components of a Direction" should "return correct values" in:
    assert(X.x == 1)
    assert(X.y == 0)
    assert(X.z == 0)
    assert(Y.x == 0)
    assert(Y.y == 1)
    assert(Y.z == 0)
    assert(Z.x == 0)
    assert(Z.y == 0)
    assert(Z.z == 1)

  "rotation" should "work around X axis" in:
    assert(X.rotate90(X) == X)
    assert(Y.rotate90(X) == Z)
    assert(Z.rotate90(X) == negY)
    assert(negX.rotate90(X) == negX)
    assert(negY.rotate90(X) == negZ)
    assert(negZ.rotate90(X) == Y)

  it should "work around Y axis" in:
    assert(X.rotate90(Y) == negZ)
    assert(Y.rotate90(Y) == Y)
    assert(Z.rotate90(Y) == X)
    assert(negX.rotate90(Y) == Z)
    assert(negY.rotate90(Y) == negY)
    assert(negZ.rotate90(Y) == negX)

  it should "work around Z axis" in:
    assert(X.rotate90(Z) == Y)
    assert(Y.rotate90(Z) == negX)
    assert(Z.rotate90(Z) == Z)
    assert(negX.rotate90(Z) == negY)
    assert(negY.rotate90(Z) == X)
    assert(negZ.rotate90(Z) == negZ)

  it should "work around -X axis" in:
    assert(X.rotate90(negX) == X)
    assert(Y.rotate90(negX) == negZ)
    assert(Z.rotate90(negX) == Y)
    assert(negX.rotate90(negX) == negX)
    assert(negY.rotate90(negX) == Z)
    assert(negZ.rotate90(negX) == negY)

  it should "work around -Y axis" in:
    assert(X.rotate90(negY) == Z)
    assert(Y.rotate90(negY) == Y)
    assert(Z.rotate90(negY) == negX)
    assert(negX.rotate90(negY) == negZ)
    assert(negY.rotate90(negY) == negY)
    assert(negZ.rotate90(negY) == X)

  it should "work around -Z axis" in:
    assert(X.rotate90(negZ) == negY)
    assert(Y.rotate90(negZ) == X)
    assert(Z.rotate90(negZ) == Z)
    assert(negX.rotate90(negZ) == Y)
    assert(negY.rotate90(negZ) == negX)
    assert(negZ.rotate90(negZ) == negZ)

  "sign" should "be correct for X" in:
    assert(X.sign == 1)
  it should "be correct for Y" in:
    assert(Y.sign == 1)
  it should "be correct for Z" in:
    assert(Z.sign == 1)
  it should "be correct for -X" in:
    assert(negX.sign == -1)
  it should "be correct for -Y" in:
    assert(negY.sign == -1)
  it should "be correct for -Z" in:
    assert(negZ.sign == -1)

  "absolute value" should "be correct for X" in:
    assert(X.abs == X)
  it should "be correct for Y" in:
    assert(Y.abs == Y)
  it should "be correct for Z" in:
    assert(Z.abs == Z)
  it should "be correct for -X" in:
    assert(negX.abs == X)
  it should "be correct for -Y" in:
    assert(negY.abs == Y)
  it should "be correct for -Z" in:
    assert(negZ.abs == Z)

  "fold1" should "be correct for X" in:
    assert(X.fold1 == -Y)
  it should "be correct for Y" in:
    assert(Y.fold1 == -Z)
  it should "be correct for Z" in:
    assert(Z.fold1 == -X)

  "fold2" should "be correct for X" in:
    assert(X.fold2 == -Z)
  it should "be correct for Y" in:
    assert(Y.fold2 == -X)
  it should "be correct for Z" in:
    assert(Z.fold2 == -Y)

