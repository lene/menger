package menger.objects

import menger.objects.Direction.{X, Y, Z, negX, negY, negZ}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DirectionSuite extends AnyFlatSpec with Matchers:

  "instantiating from valid values" should "work" in:
    Direction(1, 0, 0) should be (X)
    Direction(0, 1, 0) should be (Y)
    Direction(0, 0, 1) should be (Z)
    Direction(-1, 0, 0) should be (negX)
    Direction(0, -1, 0) should be (negY)
    Direction(0, 0, -1) should be (negZ)

  it should "work from tuple" in:
    Direction((1, 0, 0)) should be (X)
    Direction((0, 1, 0)) should be (Y)
    Direction((0, 0, 1)) should be (Z)
    Direction((-1, 0, 0)) should be (negX)
    Direction((0, -1, 0)) should be (negY)
    Direction((0, 0, -1)) should be (negZ)

  "instantiating from invalid values" should "throw IllegalArgumentException for zero vector" in:
    an [IllegalArgumentException] should be thrownBy Direction(0, 0, 0)

  it should "throw IllegalArgumentException for non-unit vectors" in:
    an [IllegalArgumentException] should be thrownBy Direction(1, 1, 0)
    an [IllegalArgumentException] should be thrownBy Direction(2, 0, 0)
    an [IllegalArgumentException] should be thrownBy Direction(-1, -1, 0)
    an [IllegalArgumentException] should be thrownBy Direction(-2, 0, 0)

  "minus on Direction" should "return its negative" in:
    -X should be (negX)
    -Y should be (negY)
    -Z should be (negZ)
    -negX should be (X)
    -negY should be (Y)
    -negZ should be (Z)

  "accessing components of a Direction" should "return correct values" in:
    X.x should be (1)
    X.y should be (0)
    X.z should be (0)
    Y.x should be (0)
    Y.y should be (1)
    Y.z should be (0)
    Z.x should be (0)
    Z.y should be (0)
    Z.z should be (1)

  "rotation" should "work around X axis" in:
    X.rotate90(X) should be (X)
    Y.rotate90(X) should be (Z)
    Z.rotate90(X) should be (negY)
    negX.rotate90(X) should be (negX)
    negY.rotate90(X) should be (negZ)
    negZ.rotate90(X) should be (Y)

  it should "work around Y axis" in:
    X.rotate90(Y) should be (negZ)
    Y.rotate90(Y) should be (Y)
    Z.rotate90(Y) should be (X)
    negX.rotate90(Y) should be (Z)
    negY.rotate90(Y) should be (negY)
    negZ.rotate90(Y) should be (negX)

  it should "work around Z axis" in:
    X.rotate90(Z) should be (Y)
    Y.rotate90(Z) should be (negX)
    Z.rotate90(Z) should be (Z)
    negX.rotate90(Z) should be (negY)
    negY.rotate90(Z) should be (X)
    negZ.rotate90(Z) should be (negZ)

  it should "work around -X axis" in:
    X.rotate90(negX) should be (X)
    Y.rotate90(negX) should be (negZ)
    Z.rotate90(negX) should be (Y)
    negX.rotate90(negX) should be (negX)
    negY.rotate90(negX) should be (Z)
    negZ.rotate90(negX) should be (negY)

  it should "work around -Y axis" in:
    X.rotate90(negY) should be (Z)
    Y.rotate90(negY) should be (Y)
    Z.rotate90(negY) should be (negX)
    negX.rotate90(negY) should be (negZ)
    negY.rotate90(negY) should be (negY)
    negZ.rotate90(negY) should be (X)

  it should "work around -Z axis" in:
    X.rotate90(negZ) should be (negY)
    Y.rotate90(negZ) should be (X)
    Z.rotate90(negZ) should be (Z)
    negX.rotate90(negZ) should be (Y)
    negY.rotate90(negZ) should be (negX)
    negZ.rotate90(negZ) should be (negZ)

  "sign" should "be correct for X" in:
    X.sign should be (1)
  it should "be correct for Y" in:
    Y.sign should be (1)
  it should "be correct for Z" in:
    Z.sign should be (1)
  it should "be correct for -X" in:
    negX.sign should be (-1)
  it should "be correct for -Y" in:
    negY.sign should be (-1)
  it should "be correct for -Z" in:
    negZ.sign should be (-1)

  "absolute value" should "be correct for X" in:
    X.abs should be (X)
  it should "be correct for Y" in:
    Y.abs should be (Y)
  it should "be correct for Z" in:
    Z.abs should be (Z)
  it should "be correct for -X" in:
    negX.abs should be (X)
  it should "be correct for -Y" in:
    negY.abs should be (Y)
  it should "be correct for -Z" in:
    negZ.abs should be (Z)

  "fold1" should "be correct for X" in:
    X.fold1 should be (-Y)
  it should "be correct for Y" in:
    Y.fold1 should be (-Z)
  it should "be correct for Z" in:
    Z.fold1 should be (-X)

  "fold2" should "be correct for X" in:
    X.fold2 should be (-Z)
  it should "be correct for Y" in:
    Y.fold2 should be (-X)
  it should "be correct for Z" in:
    Z.fold2 should be (-Y)
