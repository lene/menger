package menger.objects

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers


class GeometrySuite extends AnyFlatSpec with Matchers:
  given menger.common.ProfilingConfig = menger.common.ProfilingConfig.disabled

  "cube toString" should "return class name" in:
    Cube().toString should be("Cube")
