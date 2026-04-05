package menger.objects

import com.badlogic.gdx.math.Vector3
import menger.objects.Direction.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// Regression test for tunnel intrusion bug: negative-normal starting
// faces produced tunnel walls with inverted normals, causing level-2
// sub-tunnel geometry to shift TOWARD the tunnel center instead of away.
class TunnelIntrusionDiagnostic extends AnyFlatSpec with Matchers:
  given menger.ProfilingConfig = menger.ProfilingConfig.disabled

  private val tunnelMin = -1f / 6
  private val tunnelMax = 1f / 6
  private val tolerance = 0.001f

  private def allLevel2Faces: IndexedSeq[Face] =
    val sponge = SpongeBySurface(Vector3.Zero, 1f, level = 2f)
    val half = 0.5f
    Direction.values.toIndexedSeq.flatMap { dir =>
      val offset = half * dir.sign
      val (fx, fy, fz) = dir match
        case X | Direction.negX => (offset, 0f, 0f)
        case Y | Direction.negY => (0f, offset, 0f)
        case Z | Direction.negZ => (0f, 0f, offset)
      sponge.surfaces(Face(fx, fy, fz, 1f, dir))
    }

  private def centerStrictlyInside(
    center: Float, tMin: Float, tMax: Float
  ): Boolean =
    center > tMin + tolerance && center < tMax - tolerance

  private def faceIntrudesIntoZTunnel(face: Face): Boolean =
    centerStrictlyInside(face.xCen, tunnelMin, tunnelMax) &&
      centerStrictlyInside(face.yCen, tunnelMin, tunnelMax)

  private def faceIntrudesIntoXTunnel(face: Face): Boolean =
    centerStrictlyInside(face.yCen, tunnelMin, tunnelMax) &&
      centerStrictlyInside(face.zCen, tunnelMin, tunnelMax)

  private def faceIntrudesIntoYTunnel(face: Face): Boolean =
    centerStrictlyInside(face.xCen, tunnelMin, tunnelMax) &&
      centerStrictlyInside(face.zCen, tunnelMin, tunnelMax)

  "Level 2 SpongeBySurface faces" should
    "not have geometry inside Z tunnel" in:
      allLevel2Faces.filter(faceIntrudesIntoZTunnel) shouldBe empty

  it should "not have geometry inside X tunnel" in:
    allLevel2Faces.filter(faceIntrudesIntoXTunnel) shouldBe empty

  it should "not have geometry inside Y tunnel" in:
    allLevel2Faces.filter(faceIntrudesIntoYTunnel) shouldBe empty
