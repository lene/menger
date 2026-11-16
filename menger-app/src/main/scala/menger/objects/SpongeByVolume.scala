package menger.objects

import scala.math.abs

import menger.common.TriangleMeshData
import menger.common.Vector
import menger.common.float2string
import menger.common.x
import menger.common.y
import menger.common.z


class SpongeByVolume(
  override val center: Vector[3] = Vector.Zero[3], override val scale: Float = 1f,
  val level: Float
) extends Cube(center, scale) with FractionalLevelSponge:

  override def toTriangleMesh: TriangleMeshData = toTriangleMeshExcluding(Set.empty)

  override def toTriangleMeshExcluding(excluded: Set[Direction]): TriangleMeshData =
    if level <= 0 then super.toTriangleMeshExcluding(excluded)
    else if level.isValidInt then
      if excluded.isEmpty then getIntegerMesh else getIntegerMeshExcluding(excluded)
    else getFractionalMeshExcluding(excluded)

  private def getFractionalMeshExcluding(excluded: Set[Direction]): TriangleMeshData =
    buildFractionalMesh(
      nextLevelMesh    = SpongeByVolume(center, scale, (level + 1).floor).toTriangleMeshExcluding(excluded),
      currentLevelMesh = SpongeByVolume(center, scale, level.floor).toTriangleMeshExcluding(excluded)
    )

  private lazy val getIntegerMesh: TriangleMeshData = getIntegerMeshExcluding(Set.empty)

  /** Writes the visible faces of all leaf cubes, depth-first, into two preallocated arrays,
    * the subtrees of the top levels in parallel. It used to build a mesh per sub-cube and merge
    * them at every level (`TriangleMeshData.merge` boxes every float and concatenates indices
    * quadratically), which made level 5 slow and ran the window out of memory (usability review
    * 2026-10, session 4, F86); then it collected a `Face` object per face first, sequentially,
    * which took three quarters of the build time and held a `Face` per face next to the mesh.
    * Same faces, same order, so the same mesh. */
  private def getIntegerMeshExcluding(excluded: Set[Direction]): TriangleMeshData =
    SpongeByVolume.meshOf(center, scale, level.toInt, excluded)

  override def toString: String =
    s"SpongeByVolume(level=${float2string(level)}, ${6 * Math.pow(20, level.toInt).toLong} faces)"

object SpongeByVolume:
  /** Triangles of an integer-level mesh: 4 * 20^n + 8 * 8^n (12, 144, 2112, 36096, 672768,
    * 13062144 for levels 0 to 5). The window's slowness warning used 12 * 20^n, which counts
    * the hidden faces too (38.4M instead of 13.1M at level 5). */
  def triangleCount(level: Int): Long =
    VisibleFacesPer20 * math.pow(20, level).toLong + VisibleFacesPer8 * math.pow(8, level).toLong

  private val VisibleFacesPer20 = 4L
  private val VisibleFacesPer8 = 8L

  /** A sub-cube's position among its parent's 27 cells, and which of its 6 faces are hidden by
    * what is next to it. Directions are sets of bits: bit `d.ordinal` is direction `d`. */
  private case class SubCube(x: Int, y: Int, z: Int, neighbours: Int, inheritable: Int)

  private def bit(d: Direction): Int = 1 << d.ordinal

  /** The 20 kept sub-cubes of a level (all but the centre and the 6 face centres). In the
    * iteration order of a `Set`, as before the rewrite: it is the face order of every mesh built
    * so far. */
  private val subCubes: IndexedSeq[SubCube] =
    val filled: Set[(Int, Int, Int)] =
      (for xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1 if abs(xx) + abs(yy) + abs(zz) > 1
        yield (xx, yy, zz)).toSet
    filled.iterator.map { (xx, yy, zz) =>
      SubCube(
        xx, yy, zz,
        // a face is dropped where two kept sub-cubes touch ...
        neighbours = Direction.values.filter(d => filled.contains((xx + d.x, yy + d.y, zz + d.z)))
          .map(bit).sum,
        // ... or where the parent's face was dropped and this sub-cube lies on it
        inheritable = Direction.values.filter(d => d.x * xx + d.y * yy + d.z * zz == 1)
          .map(bit).sum
      )
    }.toIndexedSeq

  private val DirectionMasks = 1 << Direction.values.length

  /** The faces of a cube in `Cube.toTriangleMeshExcluding`'s order: direction and the face
    * centre's offset from the cube's centre, in half edges. */
  private val cubeFaceOrder: IndexedSeq[(Direction, Int, Int, Int)] = IndexedSeq(
    (Direction.X, 1, 0, 0), (Direction.negX, -1, 0, 0),
    (Direction.Y, 0, 1, 0), (Direction.negY, 0, -1, 0),
    (Direction.Z, 0, 0, 1), (Direction.negZ, 0, 0, -1)
  )

  /** For each set of hidden directions, the indices into `cubeFaceOrder` of the faces left. */
  private val visibleFaces: IndexedSeq[Array[Int]] =
    IndexedSeq.tabulate(DirectionMasks)(mask =>
      cubeFaceOrder.indices.filter(i => (mask & bit(cubeFaceOrder(i)._1)) == 0).toArray
    )

  private def directionMask(excluded: Set[Direction]): Int = excluded.iterator.map(bit).sum

  private def childMask(child: SubCube, mask: Int): Int = child.neighbours | (mask & child.inheritable)

  /** `faceCounts(n)(mask)`: the faces of a level-`n` sponge whose faces in direction set `mask`
    * are hidden. */
  private def faceCounts(maxLevel: Int): IndexedSeq[Array[Long]] =
    val cubeFaces = Array.tabulate(DirectionMasks)(mask =>
      (Direction.values.length - Integer.bitCount(mask)).toLong
    )
    (1 to maxLevel).foldLeft(IndexedSeq(cubeFaces)) { (counts, _) =>
      counts :+ Array.tabulate(DirectionMasks)(mask =>
        subCubes.map(c => counts.last(childMask(c, mask))).sum
      )
    }

  /** Subtrees with at least this many faces are written in parallel: smaller ones cost less than
    * handing them to another thread. */
  private val ParallelFromFaces = 100_000L

  /** The mesh of a level-`n` sponge: each face a quad of 4 vertices and 2 triangles, written into
    * disjoint slices of two preallocated arrays. */
  private def meshOf(
    center: Vector[3], scale: Float, n: Int, excluded: Set[Direction]
  ): TriangleMeshData =
    val counts = faceCounts(n)
    val mask = directionMask(excluded)
    val faces = counts(n)(mask)
    if faces == 0 then TriangleMeshData.empty
    else
      require(faces * Face.FloatsPerFace <= Int.MaxValue,
        s"a level $n sponge's $faces faces do not fit in one mesh array")
      val vertices = new Array[Float](faces.toInt * Face.FloatsPerFace)
      val indices = new Array[Int](faces.toInt * Face.QuadIndices.length)
      writeSubtree(vertices, indices, counts, center.x, center.y, center.z, scale, n, mask, 0)
      TriangleMeshData(vertices, indices, Face.VertexStride)

  /** Writes the faces of the level-`n` sponge at `(cx, cy, cz)`, hidden faces in `mask` left
    * out, as faces number `firstFace` on. */
  private def writeSubtree(
    vertices: Array[Float], indices: Array[Int], counts: IndexedSeq[Array[Long]],
    cx: Float, cy: Float, cz: Float, scale: Float, n: Int, mask: Int, firstFace: Int
  ): Unit =
    if n <= 0 then writeCube(vertices, indices, cx, cy, cz, scale, mask, firstFace)
    else
      val shift = scale / 3f
      val firstFaces = subCubes.scanLeft(firstFace.toLong)((first, c) =>
        first + counts(n - 1)(childMask(c, mask))
      )
      def writeChild(i: Int): Unit =
        val c = subCubes(i)
        writeSubtree(
          vertices, indices, counts,
          cx + c.x * shift, cy + c.y * shift, cz + c.z * shift,
          scale / 3f, n - 1, childMask(c, mask), firstFaces(i).toInt
        )
      if counts(n)(mask) >= ParallelFromFaces then
        java.util.stream.IntStream.range(0, subCubes.length).parallel().forEach(i => writeChild(i))
      else subCubes.indices.foreach(writeChild)

  /** A level-0 cube's visible faces, as faces number `firstFace` on. */
  private def writeCube(
    vertices: Array[Float], indices: Array[Int],
    cx: Float, cy: Float, cz: Float, scale: Float, mask: Int, firstFace: Int
  ): Unit =
    val half = scale / 2
    val visible = visibleFaces(mask)
    for k <- visible.indices do
      val (d, dx, dy, dz) = cubeFaceOrder(visible(k))
      val face = firstFace + k
      Face.writeQuad(
        vertices, face * Face.FloatsPerFace, cx + dx * half, cy + dy * half, cz + dz * half,
        scale, d
      )
      val firstVertex = face * Face.VerticesPerFace
      for q <- Face.QuadIndices.indices do
        indices(face * Face.QuadIndices.length + q) = firstVertex + Face.QuadIndices(q)
