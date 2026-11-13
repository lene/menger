package menger.engines.scene

import menger.common.TriangleMeshData
import menger.common.Vector
import menger.dsl.ResourceLimits
import menger.failure.Fail
import menger.objects.higher_d.IndexedMeshN
import menger.objects.higher_d.NdSponge
import menger.objects.higher_d.NdSpongeVariant
import menger.spec.ObjectSpec

/** What the engines draw of an `n-sponge-volume` or `n-sponge-surface` spec (menger#166): the
  * nD Menger sponge of the spec's dimension, level and `maxCentreCoordinates`, seen through the
  * spec's [[menger.spec.ProjectionSpec]] on the CPU path.
  *
  * Building the sponge is the expensive step and does not depend on the view, so the unprojected
  * mesh is memoised: an animation that turns, scales or re-views the sponge every frame builds
  * it once and only projects it again. */
object NdSpongeScene:

  // variant, dimension, whole level, k, whether it is the hole-cap mesh
  private type Key = (NdSpongeVariant, Int, Int, Int, Boolean)

  // Bounded by the faces retained, not by a count of entries: a scene of several sponges, or
  // an animation through levels 0 to 2 and their hole caps, keeps every mesh it re-projects,
  // and a few huge sponges cannot hold more than a scene's triangle budget in faces.
  private val memo = FaceBoundedMemo[Key, IndexedMeshN](
    () => ResourceLimits.sceneMaxTriangles, _.faceCount.toLong
  )

  /** The faces the memo holds now. */
  private[scene] def retainedFaces: Long = memo.retainedWeight

  def variantOf(spec: ObjectSpec): NdSpongeVariant =
    NdSpongeVariant.fromObjectType(spec.objectType).getOrElse(
      Fail.internal(s"${spec.objectType} is not an n-sponge type")
    )

  def maxCentreCoordinatesOf(spec: ObjectSpec): Int =
    spec.maxCentreCoordinates.getOrElse(NdSponge.DefaultMaxCentreCoordinates)

  /** The level the mesh is built at: a fractional level is drawn as its two whole levels (or
    * a level and its caps) by the scene builder, one mesh at a time. */
  def wholeLevelOf(spec: ObjectSpec): Int =
    spec.level.getOrElse(Fail.configuration(s"${spec.objectType} needs a level")).floor.toInt

  /** The sponge's faces before projection, or with `holeCaps` the caps of its holes: from the
    * memo when a recent call built them. */
  def mesh(spec: ObjectSpec, holeCaps: Boolean): IndexedMeshN =
    val n = NCubeScene.dimensionOf(spec)
    val key: Key =
      (variantOf(spec), n, wholeLevelOf(spec), maxCentreCoordinatesOf(spec), holeCaps)
    memo.getOrElseUpdate(key) {
      val sponge = NdSponge(key._1, n, key._3, key._4)
      if holeCaps then sponge.holeCaps else sponge
    }

  /** The triangles of the (hole-cap) mesh seen through the spec's view: projected afresh. */
  def triangleMesh(spec: ObjectSpec, holeCaps: Boolean): TriangleMeshData =
    val n = NCubeScene.dimensionOf(spec)
    val view = ProjectionConversion.liftedProjection(NCubeScene.viewOf(spec))
    mesh(spec, holeCaps).toTriangleMesh(
      spec.size, spec.axisScales(n), view.project(_), Vector.Zero[3]
    )
