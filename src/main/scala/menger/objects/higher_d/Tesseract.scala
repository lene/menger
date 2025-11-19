package menger.objects.higher_d

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import menger.objects.Builder
import menger.objects.Vector


case class Tesseract(
  size: Float = 1.0,
  material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Mesh4D:
  
  lazy val vertices: Seq[Vector[4]] =
    val coords = Seq(-size / 2, size / 2)
    for (
      xx <- coords; yy <- coords; zz <- coords; ww <- coords
    ) yield Vector[4](xx, yy, zz, ww)

  
  lazy val faceIndices: Seq[RectIndices] = RectIndices.fromTuples(
    ( 0, 1, 3, 2), ( 0, 1, 5, 4), ( 0, 1, 9, 8), ( 0, 2, 6, 4),
    ( 0, 2,10, 8), ( 0, 4,12, 8), ( 3, 1, 5, 7), ( 3, 1, 9,11),
    ( 5, 1, 9,13), ( 3, 2, 6, 7), ( 3, 2,10,11), (10, 2, 6,14),
    (15, 7, 3,11), ( 5, 4, 6, 7), (12, 4, 5,13), (12, 4, 6,14),
    (15, 7, 5,13), (15, 7, 6,14), (10, 8, 9,11), (12, 8, 9,13),
    (12, 8,10,14), (15,11, 9,13), (15,11,10,14), (15,13,12,14)
  )

  lazy val faces: Seq[Face4D] = faceIndices.map { _.toFace4D(vertices) }

  lazy val edges: Seq[(Vector[4], Vector[4])] = faces.flatMap {
    // sets instead of tuples so edges are equal regardless of direction
    case Face4D(a, b, c, d) => Seq(Set(a, b), Set(b, c), Set(c, d), Set(d, a))
  }.distinct.map(set => (set.head, set.last))


