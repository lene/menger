package menger.engines.scene

import java.util.LinkedHashMap

import scala.annotation.tailrec
import scala.jdk.CollectionConverters._

/** A least-recently-used memo whose entries are weighed (here by the faces a mesh holds) and
  * evicted, oldest first, until the retained weight is within `limit`. The newest entry stays
  * even when it alone is over the limit. `limit` is read on every insert, so a limit that
  * follows a configurable budget (menger#179) is picked up as it changes. */
final class FaceBoundedMemo[K, V](limit: () => Long, weight: V => Long):

  @SuppressWarnings(Array("org.wartremover.warts.Null"))
  private val entries = new LinkedHashMap[K, V](16, 0.75f, true)

  def getOrElseUpdate(key: K)(build: => V): V =
    entries.synchronized {
      Option(entries.get(key)).getOrElse {
        val built = build
        entries.put(key, built)
        evict()
        built
      }
    }

  /** The summed weight of the retained entries. */
  def retainedWeight: Long = entries.synchronized(entries.values.asScala.map(weight).sum)

  def size: Int = entries.synchronized(entries.size)

  @tailrec
  private def evict(): Unit =
    if entries.size > 1 && entries.values.asScala.map(weight).sum > limit() then
      val _ = entries.remove(entries.keySet.iterator.next)
      evict()
