package menger.objects.higher_d

import com.badlogic.gdx.math.{Matrix4, Vector4}

extension (m: Matrix4)
  def asArray: Array[Float] = Array(
    m.`val`(Matrix4.M00), m.`val`(Matrix4.M01), m.`val`(Matrix4.M02), m.`val`(Matrix4.M03),
    m.`val`(Matrix4.M10), m.`val`(Matrix4.M11), m.`val`(Matrix4.M12), m.`val`(Matrix4.M13),
    m.`val`(Matrix4.M20), m.`val`(Matrix4.M21), m.`val`(Matrix4.M22), m.`val`(Matrix4.M23),
    m.`val`(Matrix4.M30), m.`val`(Matrix4.M31), m.`val`(Matrix4.M32), m.`val`(Matrix4.M33)
  )

  def apply(v: Vector4): Vector4 =
    val m0 = m.`val`(Matrix4.M00) * v.x + m.`val`(Matrix4.M01) * v.y + m.`val`(Matrix4.M02) * v.z + m.`val`(Matrix4.M03) * v.w
    val m1 = m.`val`(Matrix4.M10) * v.x + m.`val`(Matrix4.M11) * v.y + m.`val`(Matrix4.M12) * v.z + m.`val`(Matrix4.M13) * v.w
    val m2 = m.`val`(Matrix4.M20) * v.x + m.`val`(Matrix4.M21) * v.y + m.`val`(Matrix4.M22) * v.z + m.`val`(Matrix4.M23) * v.w
    val m3 = m.`val`(Matrix4.M30) * v.x + m.`val`(Matrix4.M31) * v.y + m.`val`(Matrix4.M32) * v.z + m.`val`(Matrix4.M33) * v.w
    val v_ = Vector4(m0, m1, m2, m3)
    v_

  def str: String =
    val bdArray = m.`val`.map("% 2.2f".format(_))
    val mArray = Array(
      Array(bdArray(Matrix4.M00), bdArray(Matrix4.M01), bdArray(Matrix4.M02), bdArray(Matrix4.M03)).mkString("|", " ", "|"),
      Array(bdArray(Matrix4.M10), bdArray(Matrix4.M11), bdArray(Matrix4.M12), bdArray(Matrix4.M13)).mkString("|", " ", "|"),
      Array(bdArray(Matrix4.M20), bdArray(Matrix4.M21), bdArray(Matrix4.M22), bdArray(Matrix4.M23)).mkString("|", " ", "|"),
      Array(bdArray(Matrix4.M30), bdArray(Matrix4.M31), bdArray(Matrix4.M32), bdArray(Matrix4.M33)).mkString("|", " ", "|")
    )
    mArray.mkString("\n")
