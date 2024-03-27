package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4

import scala.math.abs

class TesseractSponge(level: Int) extends Mesh4D:
  lazy val faces: Seq[RectVertices4D] =
    if level == 0 then Tesseract().faces else nestedFaces.flatten

  private def nestedFaces = for (
    xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1; ww <- -1 to 1
    if abs(xx) + abs(yy) + abs(zz) + abs(ww) > 2
  ) yield shrunkSubSponge.map(translate(_, Vector4(xx / 3f, yy / 3f, zz / 3f, ww / 3f)))

  private def shrunkSubSponge: Seq[RectVertices4D] =
    subSponge.map { case (a, b, c, d) => (a / 3, b / 3, c / 3, d / 3) }

  private def subSponge: Seq[RectVertices4D] = TesseractSponge(level - 1).faces

  private def translate(face: RectVertices4D, delta: Vector4): RectVertices4D =
    face match { case (a, b, c, d) => (a + delta, b + delta, c + delta, d + delta) }


class TesseractSponge2(level: Int, size: Float = 1) extends Mesh4D:
  lazy val faces: Seq[RectVertices4D] =
    if level == 0 then Tesseract(size).faces else nestedFaces.flatten

  def nestedFaces: Seq[Seq[RectVertices4D]] =
    TesseractSponge2(level - 1).faces.map(subdividedFace)

  def subdividedFace(face: RectVertices4D): Seq[RectVertices4D] =
    def subdivideFlatParts(face: RectVertices4D): Seq[RectVertices4D] =
      // split the face into 9 smaller squares and return all except the center one
      val corners = cornerPoints(face)
      Seq(
        (corners("a"), corners("ab1"), corners("da2bc11"), corners("da2")), // 1 // top left
        (corners("ab1"), corners("ab2"), corners("da2bc12"), corners("da2bc11")), // 2 // top middle
        (corners("ab2"), corners("b"), corners("bc1"), corners("da2bc12")), // 3 // top right
        (corners("da2"), corners("da2bc11"), corners("da1bc21"), corners("da1")), // 4 // middle left
        (corners("da2bc12"), corners("bc1"), corners("bc2"), corners("da1bc22")), // 5 // middle right
        (corners("da1"), corners("da1bc21"), corners("cd2"), corners("d")), // 6 // bottom left
        (corners("da1bc21"), corners("da1bc22"), corners("cd1"), corners("cd2")), // 7 // bottom middle
        (corners("da1bc22"), corners("bc2"), corners("c"), corners("cd1")) // 8 // bottom right
      )
    def subdividePerpendicularParts(face: RectVertices4D): Seq[RectVertices4D] =
      // for each edge of the central part of the face:
      // 1. rotate the opposite vertex around the edge in 1 normal direction of the face
      // 2. rotate the opposite vertex around the edge in the other normal direction of the face
      Seq()
    subdivideFlatParts(face) ++ subdividePerpendicularParts(face)

  def rotate(v: Vector4, axis: Vector4, angle: Float): Vector4 =
    /*
    The matrix of a proper rotation R by angle θ around the axis u = (ux, uy, uz), a unit vector,
    is given by:
          |    cos θ + ux^2(1 − cos θ)   uxuy(1 − cos θ) − uz sin θ   uxuz(1 − cos θ) + uy sin θ |
      R = | uyux(1 − cos θ) + uz sin θ   cos θ + uy^2(1 − cos θ)      uyuz(1 − cos θ) − ux sin θ |
          | uzux(1 − cos θ) − uy sin θ   uzuy(1 − cos θ) + ux sin θ   cos θ + uz^2(1 − cos θ)    |
    where cos θ is the cosine of the angle θ, sin θ is the sine of the angle θ, 
    and ux, uy, and uz are the components of the unit vector u.

    This can be written more concisely as
      R = cos θ I + sin θ [u]_× + (1 − cos θ) (u ⊗ u),
    where [u]_× is the cross product matrix of u; the expression u ⊗ u is the outer product, 
    and I is the identity matrix.
                     | ux2  uxuy  uxuz |
      u ⊗ u = u uT = | uyux uy2   uyuz |
                     | uzux uzuy  uz2  |
    The cross product matrix [u]_× is defined as
                |  0  −uz   uy |
        [u]_× = | uz    0  −ux |
                | −uy   ux   0 |
    (https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle)
    
      template <> Rotation<4>::operator Matrix<4>() const {
      Matrix<4> Rxy = Matrix<4> (0, 1, axis[0]), Rxz = Matrix<4> (0, 2, axis[1]),
                Rxw = Matrix<4> (0, 3, axis[2]),  Ryz = Matrix<4> (1, 2, axis[3]),
                Ryw = Matrix<4> (1, 3, axis[4]), Rzw = Matrix<4> (2, 3, axis[5]),
                Rxyz = Rxy*Rxz, Rxwyz = Rxw*Ryz, Ryzw = Ryw*Rzw,
                Rot = Rxyz*Rxwyz*Ryzw;
      (https://github.com/lene/HyperspaceExplorer/blob/038b73b15e9462f015fb41a085ec6849ae0a6037/src/VecMath/Rotation.C#L59)
    
      template<unsigned D, typename N>
      Matrix<D, N>::Matrix (unsigned ii, unsigned jj, N theta) {
        N c = cos (theta*pi/180.), s = sin (theta*pi/180.);
        for (unsigned i = 0; i < D; i++) {          //  i: row
          for (unsigned j = 0; j < D; j++) {       //  j: col
            _M[i][j] = 0;
          }
          _M[i][i] = 1;
        }
        _M[ii][ii] =  _M[jj][jj] = c;
        _M[ii][jj] = -s;
        _M[jj][ii] = s;
      }
    (https://github.com/lene/HyperspaceExplorer/blob/038b73b15e9462f015fb41a085ec6849ae0a6037/src/VecMath/Matrix.impl.h#L44)

    Rotation of a point in 3 dimensional space by theta about an arbitrary axes defined by a line 
    between two points P1 = (x1,y1,z1) and P2 = (x2,y2,z2) can be achieved by the following steps    
    (1) translate space so that the rotation axis passes through the origin
    (2) rotate space about the x axis so that the rotation axis lies in the xz plane
    (3) rotate space about the y axis so that the rotation axis lies along the z axis
    (4) perform the desired rotation by theta about the z axis
    (5) apply the inverse of step (3)
    (6) apply the inverse of step (2)
    (7) apply the inverse of step (1)
    (https://paulbourke.net/geometry/rotate/)
    
    This can be shortened to:
    (1) translate the space so that the rotation axis passes through the origin
    (2) rotate about u axis as in the above formula
    (3) apply the inverse of step (1)


    */
    ???
    
  def cornerPoints(face: RectVertices4D): Map[String, Vector4] =
    val (a, b, c, d) = face
    val ab1 = a + (b - a) / 3
    val ab2 = a + (b - a) * 2 / 3
    val bc1 = b + (c - b) / 3
    val bc2 = b + (c - b) * 2 / 3
    val cd1 = c + (d - c) / 3
    val cd2 = c + (d - c) * 2 / 3    // reversed direction
    val da1 = d + (a - d) / 3
    val da2 = d + (a - d) * 2 / 3    // reversed direction
    val da1bc21 = da1 + (bc2 - da1) / 3
    val da1bc22 = da1 + (bc2 - da1) * 2 / 3
    val da2bc11 = da2 + (bc1 - da2) / 3
    val da2bc12 = da2 + (bc1 - da2) * 2 / 3
    Map(
      "a" -> a, "b" -> b, "c" -> c, "d" -> d,
      "ab1" -> ab1, "ab2" -> ab2, "bc1" -> bc1, "bc2" -> bc2,
      "cd1" -> cd1, "cd2" -> cd2, "da1" -> da1, "da2" -> da2,
      "da1bc21" -> da1bc21, "da1bc22" -> da1bc22, "da2bc11" -> da2bc11, "da2bc12" -> da2bc12
    )
