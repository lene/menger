package menger.objects.higher_d

import menger.objects.{Vector, Matrix}
import menger.Const
import org.scalatest.matchers.{MatchResult, Matcher}

trait CustomMatchers:
  class VectorsRoughlyEqualMatcher(expected: Vector[4, Float]) extends Matcher[Vector[4, Float]]:

    def apply(left: Vector[4, Float]): MatchResult =
      MatchResult(
        left.dst(expected) < Const.epsilon,
        s"""${left.asString} is not epsilon-equal to ${expected.asString}""",
        s"""${left.asString} equal to ${expected.asString} to ${Const.epsilon}"""
      )

  def epsilonEqual(expected: Vector[4, Float]) = new VectorsRoughlyEqualMatcher(expected)

  class MatricesRoughlyEqualMatcher(expected: Matrix[4, Float]) extends Matcher[Matrix[4, Float]]:

    def apply(left: Matrix[4, Float]): MatchResult = {
      MatchResult(
        expected.asArray.zip(left.asArray).forall(
          (expectedElement, leftElement) => expectedElement - leftElement < Const.epsilon
        ),
        s"""${left.str} is not epsilon-equal to \n${expected.str}""",
        s"""${left.str} equal to ${expected.str} to ${Const.epsilon}"""
      )
    }

  def epsilonEqual(expected: Matrix[4, Float]) = new MatricesRoughlyEqualMatcher(expected)

  def containsEpsilon(left: Iterable[Vector[4, Float]], expected: Vector[4, Float]): Boolean =
    left.exists(_.dst(expected) < Const.epsilon)

  class ContainerContainsVector4Matcher(expected: Vector[4, Float]) extends Matcher[Iterable[Vector[4, Float]]]:
    def apply(left: Iterable[Vector[4, Float]]): MatchResult =
      MatchResult(
        containsEpsilon(left, expected),
        s"""${left.map(_.asString)} does not contain ${expected.asString}""",
        s"""${left.map(_.asString)} contains ${expected.asString} to ${Const.epsilon}"""
      )

  def containEpsilon(expected: Vector[4, Float]) = new ContainerContainsVector4Matcher(expected)

  class FaceContainsAllVector4Matcher(expected: Iterable[Vector[4, Float]]) extends Matcher[Seq[Face4D]]:
    def apply(left: Seq[Face4D]): MatchResult =
      MatchResult(
        containsAllEpsilon(left, expected.toSeq),
        s"""${left.map(_.toString)} does not contain all ${expected.map(_.asString)}""",
        s"""${left.map(_.toString)} contains all ${expected.map(_.asString)}"""
      )

  def containAllEpsilon(expected: Iterable[Vector[4, Float]]) = new FaceContainsAllVector4Matcher(expected)

  private def containsAllEpsilon(rects: Seq[Face4D], expected: Seq[Vector[4, Float]]): Boolean =
    rects.map(_.asSeq).exists(rect => rect.forall(v => containsEpsilon(expected, v)))

  class Vector4SeqsMatcher(expected: Iterable[Vector[4, Float]]) extends Matcher[Iterable[Vector[4, Float]]]:
    def apply(left: Iterable[Vector[4, Float]]): MatchResult =
      MatchResult(
        left.zip(expected).forall((v1, v2) => v1.epsilonEquals(v2)),
        s"""${left.map(_.asString)} not equal ${expected.map(_.asString)}""",
        s"""${left.map(_.asString)} equals ${expected.map(_.asString)} to ${Const.epsilon}"""
      )

  def epsilonEqual(expected: Iterable[Vector[4, Float]]) = new Vector4SeqsMatcher(expected)


object CustomMatchers extends CustomMatchers

