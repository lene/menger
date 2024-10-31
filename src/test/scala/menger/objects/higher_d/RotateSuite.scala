package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import org.scalatest.flatspec.AnyFlatSpec


class RotateSuite extends AnyFlatSpec:

  "rotating around an axis" should "move a point around an axis through the origin" in:
    val point: Vector4 = Vector4(-1, -1, -1, -1)
    val axis: (Vector4, Vector4) = (Vector4(0, 0, 0, 0), Vector4(-1, -1, 1 / 3f, -1 / 3f))
    val rotated: Vector4 = rotate(point, axis, 90)
    assert(!rotated.epsilonEquals(point))

  it should "move a point around the x axis" in:
    val point: Vector4 = Vector4(-1, -1, -1, -1)
    val axis: (Vector4, Vector4) = (Vector4(0, 0, 0, 0), Vector4(1, 0, 0, 0))
    val rotated: Vector4 = rotate(point, axis, 90)
    assert(!rotated.epsilonEquals(point))

  it should "move a point around the y axis" in:
    val point: Vector4 = Vector4(-1, -1, -1, -1)
    val axis: (Vector4, Vector4) = (Vector4(0, 0, 0, 0), Vector4(0, 1, 0, 0))
    val rotated: Vector4 = rotate(point, axis, 90)
    assert(!rotated.epsilonEquals(point))

  it should "move a point around the z axis" in:
    val point: Vector4 = Vector4(-1, -1, -1, -1)
    val axis: (Vector4, Vector4) = (Vector4(0, 0, 0, 0), Vector4(0, 0, 1, 0))
    val rotated: Vector4 = rotate(point, axis, 90)
    assert(!rotated.epsilonEquals(point))

  it should "move a point around the w axis" in:
    val point: Vector4 = Vector4(-1, -1, -1, -1)
    val axis: (Vector4, Vector4) = (Vector4(0, 0, 0, 0), Vector4(0, 0, 0, 1))
    val rotated: Vector4 = rotate(point, axis, 90)
    assert(!rotated.epsilonEquals(point))

  it should "move a point around an axis not through the origin" in:
    val point: Vector4 = Vector4(1, 1, 1, 1)
    val axis: (Vector4, Vector4) = (Vector4(-1, -1, -1 / 3f, -1 / 3f), Vector4(-1, -1, 1 / 3f, -1 / 3f))
    println(">>>>>")
    val rotated: Vector4 = rotate(point, axis, 90)
    println("<<<<<")
    assert(!rotated.epsilonEquals(point))
