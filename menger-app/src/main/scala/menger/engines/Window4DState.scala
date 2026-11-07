package menger.engines

import menger.spec.ObjectSpec
import menger.spec.Projection4DSpec

/** The 4D view state as the window title shows it (usability review 2026-10, session 5, F116,
  * menger#131): Shift+drag changed the rotation with nothing on screen saying so, and the tester
  * concluded the 4D controls were missing. Taken from the first projected object, the one the
  * controls act on with all others. */
object Window4DState:

  val Hint = "Shift+drag rotates in 4D, Shift+scroll moves eye-w"
  val NoObject = "Shift+drag: no 4D object in the scene"
  val SetByScene =
    "4D view set by the scene's ProjectionSpec (Shift+drag and Shift+scroll do not apply)"

  def describe(specs: List[ObjectSpec]): Option[String] =
    val rotatable = specs.find(s => TypeRegistry.is4DFastPathType(s.objectType) && s.isLegacyView)
    rotatable.map { spec =>
      val p = spec.projection4D.getOrElse(Projection4DSpec.default)
      f"4D xw ${p.rotXW}%.1f° yw ${p.rotYW}%.1f° zw ${p.rotZW}%.1f° eye-w ${p.eyeW}%.2f ($Hint)"
    }.orElse(
      specs.find(s => TypeRegistry.is4DFastPathType(s.objectType)).map(_ => SetByScene)
    )
