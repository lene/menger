package menger.engines.scene

import menger.common.Vector

/** Which projected edges get a tube: not one a collapsed axis left without length (menger#164). */
private[scene] object EdgeTubes:

  private val MinLength = 1e-6f

  def hasLength(from: Vector[3], to: Vector[3]): Boolean = from.dst(to) >= MinLength

  def withLength(
    endpoints: IndexedSeq[Option[(Vector[3], Vector[3])]]
  ): IndexedSeq[Option[(Vector[3], Vector[3])]] =
    endpoints.map(_.filter((from, to) => hasLength(from, to)))
