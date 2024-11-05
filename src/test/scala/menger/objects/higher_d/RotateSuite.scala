package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec


class RotateSuite extends AnyFlatSpec:
  private val axisName = Array("x", "y", "z", "w")
  private def assertRotation(point: Vector4, axis: (Vector4, Vector4), angle: Float, expected: Vector4): Unit =
    val rotate = Rotate(axis, angle)
    val rotated: Vector4 = rotate(point)
    val output = s"${rotate.transformationMatrix.str}\n${str(rotated)}"
    assert(rotated.epsilonEquals(expected), output)
  private def assertRotationsEqual(axis: (Vector4, Vector4), plane1: Int, plane2: Int): Unit =
    val point = Vector4(-1, -1, -1, -1)
    val rotateAxis = Rotate(axis, 90)
    val rotatePlane = Rotate(plane1, plane2, 90)
    val output = s"\n${axisName(axis(1).toArray.indexWhere(_ > 0))}:\n${rotateAxis.transformationMatrix.str}\n" +
      s"${axisName(plane1)}${axisName(plane2)}:\n${rotatePlane.transformationMatrix.str}"
    assert(rotateAxis(point).epsilonEquals(rotatePlane(point)), output)

  "rotating around x axis" should "move a point around the x axis" in:
    assertRotation(Vector4(-1, -1, -1, -1), (Vector4(0, 0, 0, 0), Vector4(1, 0, 0, 0)), 90, Vector4(-1, -1, 1, -1))

  it should "equal rotating around yz plane" in:
    assertRotationsEqual((Vector4(0, 0, 0, 0), Vector4(1, 0, 0, 0)), 1, 2)

  "rotating around y axis" should "move a point around the y axis" in:
    assertRotation(Vector4(-1, -1, -1, -1), (Vector4(0, 0, 0, 0), Vector4(0, 1, 0, 0)), 90, Vector4(1, -1, -1, -1))

  it should "equal rotating around negative xz plane" in :
    assertRotationsEqual((Vector4(0, 0, 0, 0), Vector4(0, 1, 0, 0)), 2, 0)

  "rotating around z axis" should "move a point around the z axis" in:
    assertRotation(Vector4(-1, -1, -1, -1), (Vector4(0, 0, 0, 0), Vector4(0, 0, 1, 0)), 90, Vector4(-1, 1, -1, -1))

  it should "equal rotating around xy plane" in :
    assertRotationsEqual((Vector4(0, 0, 0, 0), Vector4(0, 0, 1, 0)), 0, 1)

  "rotating around w axis" should "move a point around the w axis" in:
    assertRotation(Vector4(-1, -1, -1, -1), (Vector4(0, 0, 0, 0), Vector4(0, 0, 0, 1)), 90, Vector4(-1, -1, -1, 1))

  it should "print all 4D base transformation matrices" in:
    for i <- 0 to 3 do
      for j <- 0 to 3 do
        if i != j && i < j then
          val rotate = Rotate(i, j, 90)
          println(s"Rotating around ${axisName(i)}${axisName(j)} plane:")
          println(rotate.transformationMatrix.str)
          println(str(rotate(Vector4(-1, -1, -1, -1))))

  ignore should "move a point around an axis not through the origin" in:
    assertRotation(
      Vector4(1, 1, 1, 1),
      (Vector4(-1, -1, -1 / 3f, -1 / 3f), Vector4(-1, -1, 1 / 3f, -1 / 3f)),
      90, Vector4(-1, -1, -1, 1)
    )
