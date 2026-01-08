package menger.objects

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CompositeSuite extends AnyFlatSpec with Matchers:
  given menger.ProfilingConfig = menger.ProfilingConfig.disabled

  "Composite toString" should "show component geometries" in:
    val sphere = Sphere()
    val cube = Cube()
    val composite = Composite(geometries = List(sphere, cube))
    composite.toString should be("Composite(Sphere, Cube)")

