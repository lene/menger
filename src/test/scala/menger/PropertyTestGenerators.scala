package menger

import menger.common.Vector
import menger.objects.Matrix
import org.scalacheck.Arbitrary
import org.scalacheck.Gen

object PropertyTestGenerators:

  val reasonableFloat: Gen[Float] = Gen.choose(-1000f, 1000f)

  val smallFloat: Gen[Float] = Gen.choose(-10f, 10f)

  val nonZeroFloat: Gen[Float] = Gen.oneOf(
    Gen.choose(-1000f, -0.001f),
    Gen.choose(0.001f, 1000f)
  )

  val angleGen: Gen[Float] = Gen.choose(-360f, 360f)

  val vector4Gen: Gen[Vector[4]] = for
    x <- reasonableFloat
    y <- reasonableFloat
    z <- reasonableFloat
    w <- reasonableFloat
  yield Vector[4](x, y, z, w)

  val smallVector4Gen: Gen[Vector[4]] = for
    x <- smallFloat
    y <- smallFloat
    z <- smallFloat
    w <- smallFloat
  yield Vector[4](x, y, z, w)

  val nonZeroVector4Gen: Gen[Vector[4]] = for
    x <- nonZeroFloat
    y <- nonZeroFloat
    z <- nonZeroFloat
    w <- nonZeroFloat
  yield Vector[4](x, y, z, w)

  val matrix4Gen: Gen[Matrix[4]] = for
    elements <- Gen.listOfN(16, reasonableFloat)
  yield Matrix[4](elements.toArray)

  val smallMatrix4Gen: Gen[Matrix[4]] = for
    elements <- Gen.listOfN(16, smallFloat)
  yield Matrix[4](elements.toArray)

  given Arbitrary[Vector[4]] = Arbitrary(vector4Gen)
  given Arbitrary[Matrix[4]] = Arbitrary(matrix4Gen)
