package menger.engines.scene

import menger.spec.ObjectSpec

/** What a builder owes an object with a `ProjectionSpec`: render it through the CPU chain or
  * refuse it with the object and the reason, never fall back to a 4D view (menger#162). */
object ProjectionSupport:

  /** A builder without a 4D mesh to project (GPU-traced IFS fractals, L-systems, ...) cannot
    * take a `ProjectionSpec`: the message for the first such spec. */
  def rejectedByMeshlessBuilder(specs: List[ObjectSpec]): Option[String] =
    specs.find(!_.isLegacyView).map(spec =>
      s"${spec.objectType} cannot take a ProjectionSpec: it needs a mesh type " +
        "(Tesseract, TesseractSponge or a regular polytope)"
    )

  /** A fractional-level 4D sponge is a pair of GPU-projected meshes with hole caps, which the
    * CPU chain does not build: the message for the first `ProjectionSpec` spec with one. */
  def rejectedFractionalLevel(specs: List[ObjectSpec], isFractionalSponge: ObjectSpec => Boolean)
    : Option[String] =
    specs.find(spec => !spec.isLegacyView && isFractionalSponge(spec)).map(spec =>
      s"${spec.objectType} with a ProjectionSpec needs a whole level, got " +
        s"${spec.level.getOrElse(0f)}"
    )
