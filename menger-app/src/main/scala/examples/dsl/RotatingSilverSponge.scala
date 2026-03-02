package examples.dsl

import scala.language.implicitConversions

import menger.dsl._

// Animated Menger sponge using the IAS (CubeSponge) construction method.
// The t parameter controls fractal level (0 to 3) and the Y-axis rotation
// of the sponge (one full rotation per level unit). The camera is fixed;
// the sponge grows and spins in place.
//
// Instance counts: level 3 = 8000 solid; fractional levels between 2 and 3 peak at
// 10800 (8000 solid + 2800 ghost). Level 4 (20^4 = 160000) exceeds the system limit.
//
// Usage:
//   --scene examples.dsl.RotatingSilverSponge --t 1.5 --max-instances 8000
//   --scene examples.dsl.RotatingSilverSponge --frames 30 --start-t 0 --end-t 3
//       --max-instances 11000 --save-name silver_sponge_%04d.png
object RotatingSilverSponge:
  private val MaxLevel     = 3f
  private val CameraRadius = 5f
  private val CameraHeight = 2.5f
  private val SpongeSize   = 2.0f
  private val TwoPi        = (2.0 * math.Pi).toFloat

  // Matte material for geometry inspection; change to Silver for final render
  private val Matte = Material(Color(0.8f, 0.8f, 0.8f), roughness = 1.0f, metallic = 0.0f, specular = 0.0f)

  def scene(t: Float): Scene =
    val level = math.max(0f, math.min(t, MaxLevel))
    val angle = t * TwoPi
    Scene(
      camera = Camera(
        position = Vec3(CameraRadius, CameraHeight, 0f),  // fixed viewpoint
        lookAt   = Vec3.Zero
      ),
      objects = List(
        Sponge(
          spongeType = CubeSponge,
          pos        = Vec3.Zero,
          level      = level,
          material   = Some(Matte),
          size       = SpongeSize,
          rotation   = angle
        )
      ),
      lights = List(
        Directional(direction = (1f, -1f, -1f),   intensity = 1.5f),
        Directional(direction = (-1f, -0.5f, 1f), intensity = 0.5f),
        Directional(direction = (0f, 1f, 0.5f),   intensity = 0.3f)
      ),
      plane = Some(Plane(Y at -3, color = "#303030")),
      background = Some(Color(0.05f, 0.05f, 0.05f))
    )
