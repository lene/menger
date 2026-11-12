package framebudget

import scala.language.implicitConversions

import menger.dsl._

/**
 * Benchmark scene for the dimension ladder's render budget (menger#159): a glass Menger sponge
 * whose level follows t in quarter steps, on a plinth, lit as the Sprint 38 scene is. Not an
 * example: the agent corpus does not see it. Measured with scripts/frame-budget.sh.
 *
 * Usage: scripts/frame-budget.sh BIN scripts/frame-budget/Moment3D.scala --at 4.5,5
 */
object Moment3D:
  // The scene's settings. G5 (menger#160) copies this block, and G4's final run measures G5's
  // scene with the same script. Keep Moment4D.scala's block identical.
  private val envMap = "assets/hdri/kiara_2_sunrise_4k.hdr"
  private val sun = Sun(
    direction = Vec3(-0.7423f, -0.0722f, -0.6662f),
    diameterDegrees = 0.53f,
    color = Color(1.00f, 0.86f, 0.40f)
  )
  private val plinthTopY = -0.45f
  private val objectSize = 1f
  // 0.01 above the plinth's top cap: a bottom face on the cap itself speckles the refraction.
  private val objectCentre = Vec3(0f, plinthTopY + 0.01f + objectSize / 2f, 0f)
  // Low, on the far side from the sun, 15 degrees to one side.
  private val cameraOffset = Vec3(-4.0f, -0.25f, -2.04f)
  private val levelStepsPerUnit = 4f

  val durationSeconds = 10f

  def scene(t: Float): Scene =
    Scene(
      camera = Camera(position = objectCentre + cameraOffset, lookAt = objectCentre),
      objects = List(
        Cylinder(
          p0 = Some(Vec3(0f, plinthTopY - 0.2f, 0f)),
          p1 = Some(Vec3(0f, plinthTopY, 0f)),
          radius = Some(1.8f),
          material = Some(Material.matte(Color(0.85f, 0.82f, 0.76f)))
        ),
        Sponge(
          spongeType = VolumeFilling,
          pos = objectCentre,
          level = math.round(t * levelStepsPerUnit) / levelStepsPerUnit,
          material = Some(Material.Glass),
          size = objectSize
        )
      ),
      lights = List(sun),
      envMap = Some(envMap),
      ibl = Some(IBL()),
      caustics = Some(Caustics(photonsPerIteration = 200000, iterations = 10)),
      render = Some(RenderSettings(antialiasing = true, shadows = true))
    )
