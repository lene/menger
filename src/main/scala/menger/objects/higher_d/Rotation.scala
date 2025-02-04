package menger.objects.higher_d

import com.badlogic.gdx.math.{Matrix4, Vector4}

import scala.annotation.targetName

case class Rotation(
  degreesRotXW: Float = 0, degreesRotYW: Float = 0, degreesRotZW: Float = 0
) extends RectMesh:
  val rotXW: Float = negativeAngleToPositive(degreesRotXW % 360f)
  val rotYW: Float = negativeAngleToPositive(degreesRotYW % 360f)
  val rotZW: Float = negativeAngleToPositive(degreesRotZW % 360f)
  val isZero: Boolean = rotXW == 0 && rotYW == 0 && rotZW == 0

  private lazy val Ryw = matrix(1, 3, rotYW)
  private lazy val Rzw = matrix(2, 3, rotZW)
  private lazy val rotate = matrix(0, 3, rotXW).mul(Ryw).mul(Rzw)

  def apply(point: Vector4): Vector4 = if isZero then point else multiply(rotate, point)

  def apply(points: Seq[Vector4]): Seq[Vector4] = points.map(apply)

  def apply(points: RectVertices4D): RectVertices4D = (
    apply(points._1), apply(points._2), apply(points._3), apply(points._4)
  )

  @targetName("plus")
  def +(r: Rotation): Rotation = Rotation(rotXW + r.rotXW, rotYW + r.rotYW, rotZW + r.rotZW)

  def multiply(m: Matrix4, v: Vector4): Vector4 =
    val m0 = m.`val`(Matrix4.M00) * v.x + m.`val`(Matrix4.M01) * v.y + m.`val`(Matrix4.M02) * v.z + m.`val`(Matrix4.M03) * v.w
    val m1 = m.`val`(Matrix4.M10) * v.x + m.`val`(Matrix4.M11) * v.y + m.`val`(Matrix4.M12) * v.z + m.`val`(Matrix4.M13) * v.w
    val m2 = m.`val`(Matrix4.M20) * v.x + m.`val`(Matrix4.M21) * v.y + m.`val`(Matrix4.M22) * v.z + m.`val`(Matrix4.M23) * v.w
    val m3 = m.`val`(Matrix4.M30) * v.x + m.`val`(Matrix4.M31) * v.y + m.`val`(Matrix4.M32) * v.z + m.`val`(Matrix4.M33) * v.w
    Vector4(m0, m1, m2, m3)

  def matrix(row: Int, col: Int, angle: Float): Matrix4 =
    val m = Array.fill(4, 4)(0f)
    for i <- 0 to 3 do m(i)(i) = 1f
    m(row)(row) = math.cos(angle * math.Pi / 180).toFloat
    m(row)(col) = -math.sin(angle * math.Pi / 180).toFloat
    m(col)(row) = math.sin(angle * math.Pi / 180).toFloat
    m(col)(col) = math.cos(angle * math.Pi / 180).toFloat
    Matrix4(m.flatten)


def negativeAngleToPositive(angle: Float): Float =
  if angle < 0 then 360 + angle else angle