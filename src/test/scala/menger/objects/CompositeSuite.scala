package menger.objects

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CompositeSuite extends AnyFlatSpec with Matchers:

  "Composite toString" should "show component geometries" in:
    val sphere = Sphere()
    val cube = Cube()
    val composite = Composite(geometries = List(sphere, cube))
    composite.toString should be("Composite(Sphere, Cube)")
