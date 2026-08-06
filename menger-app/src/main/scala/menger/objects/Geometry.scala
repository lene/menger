package menger.objects

import menger.common.ProfilingConfig
import menger.common.Vector

trait Geometry(center: Vector[3] = Vector.Zero[3], scale: Float = 1f):
  override def toString: String = getClass.getSimpleName

  // menger.objects may not depend on a logging framework (see ArchitecturePhase2Spec's
  // "not use file IO or logging" rule) or LibGDX (F10), so this layer has no sink to
  // report timing to; ProfilingConfig/msg are accepted for call-site source compatibility.
  inline def logTime[T](msg: String)(f: => T)(using config: ProfilingConfig): T = f

