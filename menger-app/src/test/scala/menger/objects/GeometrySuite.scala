package menger.objects

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers


class GeometrySuite extends AnyFlatSpec with Matchers:
  given menger.ProfilingConfig = menger.ProfilingConfig.disabled

  "sphere toString" should "return class name" in:
    Sphere().toString should be("Sphere")

  "cube toString" should "return class name" in:
    Cube().toString should be("Cube")
