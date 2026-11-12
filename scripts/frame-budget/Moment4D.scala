package framebudget

import scala.language.implicitConversions

import menger.dsl._
import menger.spec.Projection4DSpec

/**
 * Benchmark scene for the dimension ladder's render budget (menger#159): a glass tesseract
 * sponge (4D, projected to 3D) whose level follows t in quarter steps, on a plinth, lit as the
 * Sprint 38 scene is. Not an example: the agent corpus does not see it. Measured with
 * scripts/frame-budget.sh.
 *
 * Usage: scripts/frame-budget.sh BIN scripts/frame-budget/Moment4D.scala --at 3.5,4
 */
object Moment4D:
  // The scene's settings. G5 (menger#160) copies this block, and G4's final run measures G5's
  // scene with the same script. Keep Moment3D.scala's block identical.
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
  // t is the level. It should lie on the 0.25 grid and is valid for 0 to 5; larger values exceed
  // the level limit of the object type (tesseract-sponge-surface). The level is floored, with a margin
  // for float error, so that the 3 frames of one measurement (t, t + 1/30, t + 2/30) draw the
  // same geometry: a rounded level would step up inside them just below a half step.
  private val levelStepsPerUnit = 4f
  private val levelFloatMargin = 1e-3

  private def quantisedLevel(t: Float): Float =
    (math.floor(t * levelStepsPerUnit + levelFloatMargin) / levelStepsPerUnit).toFloat

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
        TesseractSponge(
          spongeType = SurfaceSubdividing,
          pos = objectCentre,
          level = quantisedLevel(t),
          material = Some(Material.Glass),
          size = objectSize,
          projection = Some(Projection4DSpec(eyeW = 4f, screenW = 0.8f))
        )
      ),
      lights = List(sun),
      envMap = Some(envMap),
      ibl = Some(IBL()),
      caustics = Some(Caustics(photonsPerIteration = 200000, iterations = 10)),
      // User, 2026-11-12: "Yes, acc16 now"; see optix-jni#115 (adaptive AA cost) and #116.
      render = Some(RenderSettings(antialiasing = false, shadows = true, accumulation = 16))
    )
