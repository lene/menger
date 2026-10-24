package menger.engines

import menger.ObjectSpec
import menger.Projection4DSpec
import menger.RotationProjectionParameters
import menger.common.Const

/** The 4D rotation a user adds in the window (Shift+drag), kept apart from the scene file's own
  * projections (usability review 2026-10, session 4, F96). Written into the specs, it was lost
  * on every live reload; kept as an offset, a reload applies it to the new specs again unless
  * the file changed its own projections, and the animated window can apply it to every frame.
  * Degrees, like `Projection4DSpec`. `eyeWSteps` counts Shift+scroll notches, each one raising
  * eye-w to the power `Projection.addExponent` or its inverse, as the interactive window does
  * (session 5, F116: the preview window ignored Shift+scroll). */
final case class Interactive4DOffset(
  rotXW: Float = 0f, rotYW: Float = 0f, rotZW: Float = 0f, eyeWSteps: Int = 0
):

  def plus(dXW: Float, dYW: Float, dZW: Float): Interactive4DOffset =
    copy(rotXW = rotXW + dXW, rotYW = rotYW + dYW, rotZW = rotZW + dZW)

  /** This offset plus a window event: its rotation, and a Shift+scroll notch when it has one
    * (a rotation-only event carries the default eye-w). */
  def plus(event: RotationProjectionParameters): Interactive4DOffset =
    plus(event.rotXW, event.rotYW, event.rotZW)
      .copy(eyeWSteps = eyeWSteps + math.signum(event.eyeW - Const.defaultEyeW).toInt)

  def isZero: Boolean = rotXW == 0f && rotYW == 0f && rotZW == 0f && eyeWSteps == 0

  /** `specs` with this rotation added to every projected object's projection. */
  def applyTo(specs: List[ObjectSpec]): List[ObjectSpec] =
    if isZero then specs
    else specs.map { spec =>
      if TypeRegistry.is4DFastPathType(spec.objectType) then
        val p = spec.projection4D.getOrElse(Projection4DSpec.default)
        spec.copy(projection4D = Some(p.copy(
          eyeW = scrolledEyeW(p),
          rotXW = p.rotXW + rotXW, rotYW = p.rotYW + rotYW, rotZW = p.rotZW + rotZW
        )))
      else spec
    }

  // Kept above screen-w, which Projection4DSpec requires.
  private def scrolledEyeW(p: Projection4DSpec): Float =
    if eyeWSteps == 0 then p.eyeW
    else
      val eyeW = math.pow(p.eyeW, math.pow(Interactive4DOffset.EyeWExponent, eyeWSteps)).toFloat
      math.max(eyeW, p.screenW + Interactive4DOffset.MinEyeWAboveScreenW)

object Interactive4DOffset:
  val Zero: Interactive4DOffset = Interactive4DOffset()

  // Projection.addExponent, the interactive window's step per Shift+scroll notch.
  private val EyeWExponent = 1.1
  private val MinEyeWAboveScreenW = 0.01f

  /** The projections a scene file declares, to tell whether a reload changed them. */
  def fileProjections(specs: List[ObjectSpec]): List[Option[Projection4DSpec]] =
    specs.map(_.projection4D)
