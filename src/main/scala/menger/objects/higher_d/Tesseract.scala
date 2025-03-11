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

  /**
   *  indices of the vertices to make up tesseract faces originally taken from
   *  https://github.com/lene/HyperspaceExplorer/blob/master/src/Displayable/Object/ObjectImplementations.cpp#L55
   *  correct ordering of indices to have the face normals point in the correct direction found by
   *  executing the following code from the ScalaTest runner:
   ```
   import scala.util.control.Breaks._
   "Checking possibility to subdivide correctly" should "lead to subface 1 have absolute coordinate values 1/2 or 1/6" in:
     tryAllFaceOrientations(0)

   for i <- 1 to 23 do
     it should s"lead to subface ${i + 1} have absolute coordinate values 1/2 or 1/6" in:
       tryAllFaceOrientations(i)

   private def tryAllFaceOrientations(i: Int): Unit = {
     val tess = Tesseract()
     val sponge = TesseractSponge2(0)
     breakable {
       tess.faceIndices(i).productIterator.toList.permutations.foreach {
         case List[Int] (i, j, k, l) =>
           try {
             val face = Face4D(
               tess.vertices(i), tess.vertices(j), tess.vertices(k), tess.vertices(l)
             )
             val subdividedFace = sponge.subdividedFace(face)
             val cornerPoints = subdividedFace.flatMap(_.asSeq)
             val cornerCoordinateValues = cornerPoints.toSet.flatMap(_.toArray).map(math.abs).map(round)
             if cornerCoordinateValues.diff(Set(round(1 / 2f), round(1 / 6f))).isEmpty then
               logger.info(s"found $i, $j, $k, $l")
               break
           } catch
             case _: IllegalArgumentException =>
         case _ =>
       }
     }
   }
   ```
   */
  lazy val faceIndices: Seq[RectIndices] = Seq(
    ( 0, 1, 3, 2), ( 0, 1, 5, 4), ( 0, 1, 9, 8), ( 0, 2, 6, 4),
    ( 0, 2,10, 8), ( 0, 4,12, 8), (3, 1, 5, 7), (3, 1, 9, 11),
    (5, 1, 9, 13), (3, 2, 6, 7), (3, 2, 10, 11), (10, 2, 6, 14),
    (15, 7, 3, 11), (5, 4, 6, 7), (12, 4, 5, 13), (12, 4, 6, 14),
    (15, 7, 5, 13), (15, 7, 6, 14), (10, 8, 9, 11), (12, 8, 9, 13),
    (12, 8, 10, 14), (15, 11, 9, 13), (15, 11, 10, 14), (15, 13, 12, 14)
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
