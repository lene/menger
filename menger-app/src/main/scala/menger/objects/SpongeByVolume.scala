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

  /** Collects the visible faces of all leaf cubes depth-first, then writes them into one
    * preallocated array. It used to build a mesh per sub-cube and merge them at every level
    * (`TriangleMeshData.merge` boxes every float and concatenates indices quadratically), which
    * made level 5 slow and ran the window out of memory (usability review 2026-10, session 4,
    * F86). Same faces, same order, so the same mesh. */
  private def getIntegerMeshExcluding(excluded: Set[Direction]): TriangleMeshData =
    SpongeByVolume.meshOf(SpongeByVolume.leafFaces(center, scale, level.toInt, excluded).toArray)

  override def toString: String =
    s"SpongeByVolume(level=${float2string(level)}, ${6 * Math.pow(20, level.toInt).toLong} faces)"

object SpongeByVolume:
  /** The 20 kept sub-cubes of a level (all but the centre and the 6 face centres). A `Set`, as
    * before the rewrite: its iteration order is the face order of every mesh built so far. */
  private val filledPositions: Set[(Int, Int, Int)] =
    (for xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1 if abs(xx) + abs(yy) + abs(zz) > 1
      yield (xx, yy, zz)).toSet

  /** The visible faces of a level-`n` sponge's leaf cubes, depth-first: a face is dropped where
    * two kept sub-cubes touch, or where the parent's face was dropped. */
  private def leafFaces(
    center: Vector[3], scale: Float, n: Int, excluded: Set[Direction]
  ): Iterator[Face] =
    if n <= 0 then cubeFaces(center, scale, excluded)
    else
      val shift = scale / 3f
      filledPositions.iterator.flatMap { (xx, yy, zz) =>
        val neighborExclude = Direction.values.filter { d =>
          filledPositions.contains((xx + d.x, yy + d.y, zz + d.z))
        }.toSet
        val inheritedExclude = excluded.filter(d => d.x * xx + d.y * yy + d.z * zz == 1)
        leafFaces(
          Vector[3](center.x + xx * shift, center.y + yy * shift, center.z + zz * shift),
          scale / 3f, n - 1, neighborExclude ++ inheritedExclude
        )
      }

  /** A level-0 cube's faces, in `Cube.toTriangleMeshExcluding`'s order. */
  private def cubeFaces(center: Vector[3], scale: Float, excluded: Set[Direction]): Iterator[Face] =
    val half = scale / 2
    Iterator(
      Face(center.x + half, center.y, center.z, scale, Direction.X),
      Face(center.x - half, center.y, center.z, scale, Direction.negX),
      Face(center.x, center.y + half, center.z, scale, Direction.Y),
      Face(center.x, center.y - half, center.z, scale, Direction.negY),
      Face(center.x, center.y, center.z + half, scale, Direction.Z),
      Face(center.x, center.y, center.z - half, scale, Direction.negZ)
    ).filterNot(f => excluded.contains(f.normal))

  /** One mesh of `faces`, each a quad of 4 vertices and 2 triangles, written in parallel into
    * disjoint slices of two preallocated arrays. */
  private def meshOf(faces: Array[Face]): TriangleMeshData =
    if faces.isEmpty then TriangleMeshData.empty
    else
      val vertices = new Array[Float](faces.length * Face.FloatsPerFace)
      val indices = new Array[Int](faces.length * Face.QuadIndices.length)
      java.util.stream.IntStream.range(0, faces.length).parallel().forEach { i =>
        faces(i).writeVertices(vertices, i * Face.FloatsPerFace)
        val firstVertex = i * Face.VerticesPerFace
        Face.QuadIndices.indices.foreach { k =>
          indices(i * Face.QuadIndices.length + k) = firstVertex + Face.QuadIndices(k)
        }
      }
      TriangleMeshData(vertices, indices, Face.VertexStride)

