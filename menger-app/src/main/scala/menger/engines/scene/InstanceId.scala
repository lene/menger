package menger.engines.scene

opaque type InstanceId = Int

object InstanceId:
  def fromNative(rawId: Int, operation: => String): InstanceId =
    if rawId >= 0 then rawId
    else sys.error(s"Native renderer failed to add $operation")

  def raw(instanceId: InstanceId): Int = instanceId
