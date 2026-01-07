package menger.objects

import menger.PropertyTestGenerators.matrix4Gen
import menger.PropertyTestGenerators.smallMatrix4Gen
import menger.PropertyTestGenerators.vector4Gen
import menger.common.Vector
import menger.objects.higher_d.CustomMatchers.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class MatrixPropertySuite extends AnyFlatSpec with Matchers with ScalaCheckPropertyChecks:

  private def matrixApproxEqual(m1: Matrix[4], m2: Matrix[4], tolerance: Float = 0.01f): Boolean =
    m1.m.zip(m2.m).forall((a, b) => math.abs(a - b) < tolerance)

  "Matrix multiplication" should "have identity as left neutral element" in:
    forAll(matrix4Gen) { m =>
      val identity = Matrix.identity[4]
      (identity * m) should epsilonEqual(m)
    }

  it should "have identity as right neutral element" in:
    forAll(matrix4Gen) { m =>
      val identity = Matrix.identity[4]
      (m * identity) should epsilonEqual(m)
    }

  it should "be associative for small matrices" in:
    forAll(smallMatrix4Gen, smallMatrix4Gen, smallMatrix4Gen) { (a, b, c) =>
      matrixApproxEqual((a * b) * c, a * (b * c)) shouldBe true
    }

  "Matrix-vector multiplication" should "preserve identity" in:
    forAll(vector4Gen) { v =>
      val identity = Matrix.identity[4]
      identity(v) should epsilonEqual(v)
    }

  it should "produce zero when multiplying zero matrix by any vector" in:
    forAll(vector4Gen) { v =>
      val zero = Matrix[4](Array.fill(16)(0f))
      zero(v) should epsilonEqual(Vector.Zero[4])
    }

  "Matrix equality" should "be reflexive" in:
    forAll(matrix4Gen) { m =>
      m should epsilonEqual(m)
    }

  it should "detect identity matrix" in:
    val identity = Matrix.identity[4]
    identity should epsilonEqual(Matrix.identity[4])
