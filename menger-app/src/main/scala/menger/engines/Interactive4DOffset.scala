package menger.engines

import menger.ObjectSpec
import menger.Projection4DSpec

/** The 4D rotation a user adds in the window (Shift+drag), kept apart from the scene file's own
  * projections (usability review 2026-10, session 4, F96). Written into the specs, it was lost
  * on every live reload; kept as an offset, a reload applies it to the new specs again unless
  * the file changed its own projections, and the animated window can apply it to every frame.
  * Degrees, like `Projection4DSpec`. */
final case class Interactive4DOffset(rotXW: Float = 0f, rotYW: Float = 0f, rotZW: Float = 0f):

  def plus(dXW: Float, dYW: Float, dZW: Float): Interactive4DOffset =
    Interactive4DOffset(rotXW + dXW, rotYW + dYW, rotZW + dZW)

  def isZero: Boolean = rotXW == 0f && rotYW == 0f && rotZW == 0f

  /** `specs` with this rotation added to every projected object's projection. */
  def applyTo(specs: List[ObjectSpec]): List[ObjectSpec] =
    if isZero then specs
    else specs.map { spec =>
      if TypeRegistry.is4DFastPathType(spec.objectType) then
        val p = spec.projection4D.getOrElse(Projection4DSpec.default)
        spec.copy(projection4D = Some(p.copy(
          rotXW = p.rotXW + rotXW, rotYW = p.rotYW + rotYW, rotZW = p.rotZW + rotZW
        )))
      else spec
    }

object Interactive4DOffset:
  val Zero: Interactive4DOffset = Interactive4DOffset()

  /** The projections a scene file declares, to tell whether a reload changed them. */
  def fileProjections(specs: List[ObjectSpec]): List[Option[Projection4DSpec]] =
    specs.map(_.projection4D)
