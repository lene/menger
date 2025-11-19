package menger


case class ProfilingConfig(minDurationMs: Option[Int]):
  inline def isEnabled: Boolean = minDurationMs.isDefined
  def threshold: Int = minDurationMs.getOrElse(Int.MaxValue)

object ProfilingConfig:
  val disabled: ProfilingConfig = ProfilingConfig(None)
  def enabled(minMs: Int): ProfilingConfig = ProfilingConfig(Some(minMs))
