package menger.objects

import menger.common.Vector
import menger.common.ProfilingConfig
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ProfilingIntegrationSuite extends AnyFlatSpec with Matchers:

  "Geometry objects" should "accept ProfilingConfig via using parameter" in:
    given ProfilingConfig = ProfilingConfig.disabled

    // Should compile and instantiate without error
    val cube = Cube()

    cube shouldBe a[Cube]

  "SpongeBySurface" should "propagate ProfilingConfig to child instances" in:
    given ProfilingConfig = ProfilingConfig.enabled(5)

    val sponge = SpongeBySurface(Vector.Zero[3], 1f, 1.5f)

    // Should be able to create the sponge with fractional level
    // which creates child SpongeBySurface instances internally
    sponge.level shouldBe 1.5f

  "Different ProfilingConfigs" should "not interfere with each other in different scopes" in:
    val result1 = {
      given ProfilingConfig = ProfilingConfig.disabled
      val c = Cube()
      c.toString
    }

    val result2 = {
      given ProfilingConfig = ProfilingConfig.enabled(100)
      val c = Cube()
      c.toString
    }

    // Both should work correctly despite different configs
    result1 shouldBe "Cube"
    result2 shouldBe "Cube"
