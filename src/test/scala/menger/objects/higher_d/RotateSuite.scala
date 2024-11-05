package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec


class RotateSuite extends AnyFlatSpec:
  private val axisName = Array("x", "y", "z", "w")

  "it" should "be possible to print all 4D base transformation matrices" in:
    val axisNames = Array("x", "y", "z", "w")
    for i <- 0 to 3 do
      for j <- 0 to 3 do
        if  i < j then
          val rotate = Rotate(Plane(i, j), 90)
          println(s"Rotating around ${axisNames(i)}${axisNames(j)} plane:")
          println(rotate.transformationMatrix.str)

  private def checkPlaneAndLineRotation(
      point: Vector4, plane: Plane, direction: (Vector4, Vector4), expected1: Vector4, expected2: Vector4
  ): Unit =
    val rotate = Rotate(plane, direction, 90)
    assert(rotate.length == 2, s"\n${rotate.mkString(", ")} size != 2")
    val rotated = rotate.map(_.apply(point))
    assert(
      rotated(0).epsilonEquals(expected1),
      s"\n${rotate(0).transformationMatrix.str}\n${str(rotated(0))} != ${str(expected1)}\n"
    )
    assert(
      rotated(1).epsilonEquals(expected2),
      s"\n${rotate(1).transformationMatrix.str}\n${str(rotated(1))} != ${str(expected2)}\n"
    )
  
  "rotating around a plane and an axis" should "be correct for rotating around x in the xy plane" in:
    checkPlaneAndLineRotation(
      Vector4(0, 1, 0, 0),
      Plane(0, 1), (Vector4(0, 0, 0, 0), Vector4(1, 0, 0, 0)),
      Vector4(0, 0, -1, 0), Vector4(0, 0, 0, -1)
    )

  it should "be correct for rotating around y in the xy plane" in:
    checkPlaneAndLineRotation(
      Vector4(1, 0, 0, 0),
      Plane(0, 1), (Vector4(0, 0, 0, 0), Vector4(0, 1, 0, 0)),
      Vector4(0, 0, -1, 0), Vector4(0, 0, 0, -1)
    )
  
  it should "fail for rotating around z in the xy plane" in:
    assertThrows[IllegalArgumentException] {
      Rotate(Plane(0, 1), (Vector4(0, 0, 0, 0), Vector4(0, 0, 1, 0)), 90)
    }

  it should "fail for rotating around w in the xy plane" in:
    assertThrows[IllegalArgumentException] {
      Rotate(Plane(0, 1), (Vector4(0, 0, 0, 0), Vector4(0, 0, 0, 1)), 90)
    }

  it should "be correct for rotating around x in the xz plane" in:
    checkPlaneAndLineRotation(
      Vector4(0, 0, 1, 0),
      Plane(0, 2), (Vector4(0, 0, 0, 0), Vector4(1, 0, 0, 0)),
      Vector4(0, -1, 0, 0), Vector4(0, 0, 0, -1)
    )

  it should "be correct for rotating around z in the xz plane" in:
    checkPlaneAndLineRotation(
      Vector4(1, 0, 0, 0),
      Plane(0, 2), (Vector4(0, 0, 0, 0), Vector4(0, 0, 1, 0)),
      Vector4(0, -1, 0, 0), Vector4(0, 0, 0, -1)
    )
  it should "fail for rotating around y in the xz plane" in:
    assertThrows[IllegalArgumentException] {
      Rotate(Plane(0, 2), (Vector4(0, 0, 0, 0), Vector4(0, 1, 0, 0)), 90)
    }
  it should "fail for rotating around w in the xz plane" in:
    assertThrows[IllegalArgumentException] {
      Rotate(Plane(0, 2), (Vector4(0, 0, 0, 0), Vector4(0, 0, 0, 1)), 90)
    }
  it should "be correct for rotating around x in the xw plane" in:
    checkPlaneAndLineRotation(
      Vector4(0, 0, 0, 1),
      Plane(0, 3), (Vector4(0, 0, 0, 0), Vector4(1, 0, 0, 0)),
      Vector4(0, -1, 0, 0), Vector4(0, 0, -1, 0)
    )
  it should "be correct for rotating around w in the xw plane" in:
    checkPlaneAndLineRotation(
      Vector4(1, 0, 0, 0),
      Plane(0, 3), (Vector4(0, 0, 0, 0), Vector4(0, 0, 0, 1)),
      Vector4(0, -1, 0, 0), Vector4(0, 0, -1, 0)
    )
  it should "fail for rotating around y in the xw plane" in:
    assertThrows[IllegalArgumentException] {
      Rotate(Plane(0, 3), (Vector4(0, 0, 0, 0), Vector4(0, 1, 0, 0)), 90)
    }
  it should "fail for rotating around z in the xw plane" in:
    assertThrows[IllegalArgumentException] {
      Rotate(Plane(0, 3), (Vector4(0, 0, 0, 0), Vector4(0, 0, 1, 0)), 90)
    }
  it should "be correct for rotating around y in the yz plane" in:
    checkPlaneAndLineRotation(
      Vector4(0, 1, 0, 0),
      Plane(1, 2), (Vector4(0, 0, 0, 0), Vector4(0, 0, 1, 0)),
      Vector4(-1, 0, 0, 0), Vector4(0, 0, 0, -1)
    )
  it should "be correct for rotating around z in the yz plane" in:
    checkPlaneAndLineRotation(
      Vector4(0, 0, 1, 0),
      Plane(1, 2), (Vector4(0, 0, 0, 0), Vector4(0, 1, 0, 0)),
      Vector4(-1, 0, 0, 0), Vector4(0, 0, 0, -1)
    )
  it should "fail for rotating around x in the yz plane" in:
    assertThrows[IllegalArgumentException] {
      Rotate(Plane(1, 2), (Vector4(0, 0, 0, 0), Vector4(1, 0, 0, 0)), 90)
    }
  it should "fail for rotating around w in the yz plane" in:
    assertThrows[IllegalArgumentException] {
      Rotate(Plane(1, 2), (Vector4(0, 0, 0, 0), Vector4(0, 0, 0, 1)), 90)
    }
  it should "be correct for rotating around y in the yw plane" in:
    checkPlaneAndLineRotation(
      Vector4(0, 0, 0, 1),
      Plane(1, 3), (Vector4(0, 0, 0, 0), Vector4(0, 1, 0, 0)),
      Vector4(-1, 0, 0, 0), Vector4(0, 0, -1, 0)
    )
  it should "be correct for rotating around w in the yw plane" in:
    checkPlaneAndLineRotation(
      Vector4(0, 0, 1, 0),
      Plane(1, 3), (Vector4(0, 0, 0, 0), Vector4(0, 0, 0, 1)),
      Vector4(-1, 0, 0, 0), Vector4(0, 0, -1, 0)
    )