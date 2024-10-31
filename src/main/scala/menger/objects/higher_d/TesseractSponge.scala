package menger.objects.higher_d

import com.badlogic.gdx.math.{Matrix4, Vector4}

import scala.math.abs

class TesseractSponge(level: Int) extends Mesh4D:
  assert(level >= 0, "Level must be non-negative")
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
  assert(level >= 0, "Level must be non-negative")
  lazy val faces: Seq[RectVertices4D] =
    if level == 0 then Tesseract(size).faces else nestedFaces.flatten

  def nestedFaces: Seq[Seq[RectVertices4D]] =
    TesseractSponge2(level - 1).faces.map(subdividedFace)

  def subdividedFace(face: RectVertices4D): Seq[RectVertices4D] =
    subdivideFlatParts(face) ++ subdividePerpendicularParts(face)

  def subdivideFlatParts(face: RectVertices4D): Seq[RectVertices4D] =
    // split the face into 9 smaller squares and return all except the center one
    val c = cornerPoints(face)
    Seq(
      (c("a"), c("ab1"), c("da2bc11"), c("da2")), // 1 // top left
      (c("ab1"), c("ab2"), c("da2bc12"), c("da2bc11")), // 2 // top middle
      (c("ab2"), c("b"), c("bc1"), c("da2bc12")), // 3 // top right
      (c("da2"), c("da2bc11"), c("da1bc21"), c("da1")), // 4 // middle left
      (c("da2bc12"), c("bc1"), c("bc2"), c("da1bc22")), // 5 // middle right
      (c("da1"), c("da1bc21"), c("cd2"), c("d")), // 6 // bottom left
      (c("da1bc21"), c("da1bc22"), c("cd1"), c("cd2")), // 7 // bottom middle
      (c("da1bc22"), c("bc2"), c("c"), c("cd1")) // 8 // bottom right
    )

  def subdividePerpendicularParts(face: RectVertices4D): Seq[RectVertices4D] =
    // for each edge of the central part of the face:
    // 1. rotate the opposite vertex around the edge in 1 normal direction of the face
    // 2. rotate the opposite vertex around the edge in the other normal direction of the face
    val c = cornerPoints(face)
    val centralPart = (c("da2bc11"), c("da2bc12"), c("da1bc22"), c("da1bc21"))
    val edges = Seq(
      (centralPart(0), centralPart(1)), (centralPart(1), centralPart(2)),
      (centralPart(2), centralPart(3)), (centralPart(3), centralPart(0))
    )
    val oppositeEdges = edges.drop(2) ++ edges.take(2)

    val rotatedOneWay = for (i <- edges.indices) yield
      (edges(i)(0), edges(i)(1),
        rotate(oppositeEdges(i)(0), edges(i), 90), rotate(oppositeEdges(i)(1), edges(i), 90)
      )
    val rotatedOtherWay = for (i <- edges.indices) yield
      (edges(i)(0), edges(i)(1),
        rotate(oppositeEdges(i)(0), edges(i), 90), rotate(oppositeEdges(i)(1), edges(i), 90)
      )
    rotatedOneWay ++ rotatedOtherWay

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

extension (m: Matrix4)
  def add(m1: Matrix4): Matrix4 =
    val mArray = Array(
      m.`val`(Matrix4.M00), m.`val`(Matrix4.M01), m.`val`(Matrix4.M02), m.`val`(Matrix4.M03),
      m.`val`(Matrix4.M10), m.`val`(Matrix4.M11), m.`val`(Matrix4.M12), m.`val`(Matrix4.M13),
      m.`val`(Matrix4.M20), m.`val`(Matrix4.M21), m.`val`(Matrix4.M22), m.`val`(Matrix4.M23),
      m.`val`(Matrix4.M30), m.`val`(Matrix4.M31), m.`val`(Matrix4.M32), m.`val`(Matrix4.M33)
    )
    val m1Array = Array(
      m1.`val`(Matrix4.M00), m1.`val`(Matrix4.M01), m1.`val`(Matrix4.M02), m1.`val`(Matrix4.M03),
      m1.`val`(Matrix4.M10), m1.`val`(Matrix4.M11), m1.`val`(Matrix4.M12), m1.`val`(Matrix4.M13),
      m1.`val`(Matrix4.M20), m1.`val`(Matrix4.M21), m1.`val`(Matrix4.M22), m1.`val`(Matrix4.M23),
      m1.`val`(Matrix4.M30), m1.`val`(Matrix4.M31), m1.`val`(Matrix4.M32), m1.`val`(Matrix4.M33)
    )
    Matrix4(mArray.zip(m1Array).map((a, b) => a + b))
  def multiply(v: Vector4): Vector4 =
//    println(s"v: $v")
    val m0 = m.`val`(Matrix4.M00) * v.x + m.`val`(Matrix4.M01) * v.y + m.`val`(Matrix4.M02) * v.z + m.`val`(Matrix4.M03) * v.w
    val m1 = m.`val`(Matrix4.M10) * v.x + m.`val`(Matrix4.M11) * v.y + m.`val`(Matrix4.M12) * v.z + m.`val`(Matrix4.M13) * v.w
    val m2 = m.`val`(Matrix4.M20) * v.x + m.`val`(Matrix4.M21) * v.y + m.`val`(Matrix4.M22) * v.z + m.`val`(Matrix4.M23) * v.w
    val m3 = m.`val`(Matrix4.M30) * v.x + m.`val`(Matrix4.M31) * v.y + m.`val`(Matrix4.M32) * v.z + m.`val`(Matrix4.M33) * v.w
    val v_ = Vector4(m0, m1, m2, m3)
//    println(s"v': $v_")
    v_

def rotate(point: Vector4, axis: (Vector4, Vector4), angle: Float): Vector4 =
  /*
  In 3D the matrix of a proper rotation R by angle θ around the axis u = (ux, uy, uz),
  a unit vector, is given by:
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
  */
  /*
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
  val u = (axis(1) - axis(0)).nor()
  val uxu = outerProduct(u)
  val ux = crossProductMatrix(u)
  val I = Matrix4().idt()
  val cosTheta = math.cos(angle.toRadians).toFloat
  val sinTheta = math.sin(angle.toRadians).toFloat
  val transformationMatrix = I.scl(cosTheta).add(ux.scl(sinTheta)).add(uxu.scl(1 - cosTheta))
  transformationMatrix.multiply(point - axis(0)) + axis(0)

def outerProduct(u: Vector4): Matrix4 =
  val (x, y, z, w) = (u.x, u.y, u.z, u.w)
  Matrix4(Array(
    x * x, x * y, x * z, x * w,
    y * x, y * y, y * z, y * w,
    z * x, z * y, z * z, z * w,
    w * x, w * y, w * z, w * w
  ))

def crossProductMatrix(u: Vector4): Matrix4 =
  val (x, y, z, w) = (u.x, u.y, u.z, u.w)
  Matrix4(Array(
    0, -z, y, 0,
    z, 0, -x, 0,
    -y, x, 0, 0,
    0, 0, 0, 0
  ))
