package menger.objects.higher_d

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.math.Vector4
import menger.objects.*

/** A tesseract of edge length `size` centered at the origin */
case class Tesseract(
  size: Float = 1.0,
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Mesh4D:

  lazy val vertices: Seq[Vector4] = for (
    xx <- -1 to 1 by 2; yy <- -1 to 1 by 2; zz <- -1 to 1 by 2; ww <- -1 to 1 by 2
  ) yield Vector4(xx * size / 2, yy * size / 2, zz * size / 2, ww * size / 2)

  // order of vertices originally taken from
  // https://github.com/lene/HyperspaceExplorer/blob/master/src/Displayable/Object/ObjectImplementations.cpp#L55
  // sorting fixed to keep the face normals correct
  lazy val faceIndices: Seq[RectIndices] = Seq(
    ( 0, 1, 3, 2), ( 0, 1, 5, 4), ( 0, 1, 9, 8), ( 0, 2, 6, 4),
    ( 0, 2,10, 8), ( 0, 4,12, 8), ( 1, 3, 7, 5), ( 1, 3,11, 9),
    ( 1, 5,13, 9), ( 2, 3, 7, 6), ( 2, 3,11,10), ( 2, 6,14,10),
    (11,15, 7, 3), ( 4, 5, 7, 6), ( 4, 5,13,12), ( 4, 6,14,12),
    (13,15, 7, 5), (14,15, 7, 6), ( 8, 9,11,10), ( 8, 9,13,12),
    ( 8,10,14,12), (13,15,11, 9), (14,15,11,10), (14,15,13,12)
  )

  lazy val faces: Seq[Face4D] = faceIndices.map {
    case (a, b, c, d) => Face4D(vertices(a), vertices(b), vertices(c), vertices(d))
  }

  lazy val edges: Seq[(Vector4, Vector4)] = faces.flatMap {
    // sets instead of tuples so edges are equal regardless of direction
    case Face4D(a, b, c, d) => Seq(Set(a, b), Set(b, c), Set(c, d), Set(d, a))
  }.distinct.map(set => (set.head, set.last))

/** template for moving indices around, remove this once no longer used - negative numbers contain crossed edges*/
val permutations = List(
  (8, 10, 14, 12), (8, 10, 12, 14), (-8, 14, 10, 12), (8, 14, 12, 10),
  (-8, 12, 10, 14), (8, 12, 14, 10), (10, 8, 14, 12), (10, 8, 12, 14),
  (-10, 14, 8, 12), (10, 14, 12, 8), (-10, 12, 8, 14), (10, 12, 14, 8),
  (14, 8, 10, 12), (-14, 8, 12, 10), (14, 10, 8, 12), (-14, 10, 12, 8),
  (14, 12, 8, 10), (14, 12, 10, 8), (12, 8, 10, 14), (-12, 8, 14, 10),
  (12, 10, 8, 14), (-12, 10, 14, 8), (12, 14, 8, 10), (12, 14, 10, 8)
)
