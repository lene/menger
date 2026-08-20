package menger

import menger.engines.TypeRegistry

/** Sprint 36 H5.1: `--rot-x/-y/-z/-x-w/-y-w/-z-w` were parsed but never applied to any scene --
  * `RotationProjectionParameters.apply(opts)`, their sole reader, had no caller. Compose them
  * into every spec's own (already-wired) rotation fields instead of introducing a new
  * mechanism: `rotation` for all objects, plus `projection4D`'s W-plane angles for 4D-fast-path
  * types. Per-object `rot-x=`/`rot-x-w=` on the same spec still apply on top of this.
  */
object GlobalRotation:
  def apply(opts: MengerCLIOptions, specs: List[ObjectSpec]): List[ObjectSpec] =
    val rx = math.toRadians(opts.rotX().toDouble).toFloat
    val ry = math.toRadians(opts.rotY().toDouble).toFloat
    val rz = math.toRadians(opts.rotZ().toDouble).toFloat
    specs.map { spec =>
      val withRotation = spec.copy(rotation = ObjectRotation(
        spec.rotation.x + rx, spec.rotation.y + ry, spec.rotation.z + rz
      ))
      if TypeRegistry.is4DFastPathType(spec.objectType) then
        val proj = withRotation.projection4D.getOrElse(Projection4DSpec())
        withRotation.copy(projection4D = Some(proj.copy(
          rotXW = proj.rotXW + opts.effectiveRotXW,
          rotYW = proj.rotYW + opts.effectiveRotYW,
          rotZW = proj.rotZW + opts.effectiveRotZW
        )))
      else withRotation
    }
