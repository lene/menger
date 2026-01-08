package menger.objects

import com.badlogic.gdx.math.Vector3
import menger.ProfilingConfig
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ProfilingIntegrationSuite extends AnyFlatSpec with Matchers:

  "Geometry objects" should "accept ProfilingConfig via using parameter" in:
    given ProfilingConfig = ProfilingConfig.disabled

    // These should compile and instantiate without error
    val square = Square()
    val cube = Cube()

    square should not be null
    cube should not be null

  "SpongeBySurface" should "propagate ProfilingConfig to child instances" in:
    given ProfilingConfig = ProfilingConfig.enabled(5)

    val sponge = SpongeBySurface(Vector3.Zero, 1f, 1.5f)

    // Should be able to create the sponge with fractional level
    // which creates child SpongeBySurface instances internally
    sponge.level shouldBe 1.5f

  "Composite" should "work with geometries having ProfilingConfig" in:
    given ProfilingConfig = ProfilingConfig.disabled

    val square = Square()
    val cube = Cube()
    val composite = Composite(geometries = List(square, cube))

    composite.toString should include("Square")
    composite.toString should include("Cube")

  "RotatedProjection" should "accept ProfilingConfig" in:
    given ProfilingConfig = ProfilingConfig.enabled(10)

    val projection = menger.objects.higher_d.RotatedProjection(
      Vector3.Zero, 1f,
      menger.objects.higher_d.Tesseract(),
      menger.objects.higher_d.Projection(4, 1)
    )

    projection should not be null

  "FractionalRotatedProjection" should "accept ProfilingConfig" in:
    given ProfilingConfig = ProfilingConfig.disabled

    val frac = menger.objects.higher_d.FractionalRotatedProjection(
      Vector3.Zero, 1f,
      (level: Float) => menger.objects.higher_d.TesseractSponge(level),
      1.5f,
      menger.objects.higher_d.Projection(4, 1)
    )

    frac.level shouldBe 1.5f

  "Different ProfilingConfigs" should "not interfere with each other in different scopes" in:
    val result1 = {
      given ProfilingConfig = ProfilingConfig.disabled
      val s = Square()
      s.toString
    }

    val result2 = {
      given ProfilingConfig = ProfilingConfig.enabled(100)
      val s = Square()
      s.toString
    }

    // Both should work correctly despite different configs
    result1 shouldBe "Square"
    result2 shouldBe "Square"
