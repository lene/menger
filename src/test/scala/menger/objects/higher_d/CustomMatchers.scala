package menger.objects.higher_d

import menger.common.Vector
import menger.objects.Matrix
import menger.common.Const
import org.scalatest.matchers.{MatchResult, Matcher}

trait CustomMatchers:
  class VectorsRoughlyEqualMatcher(expected: Vector[4]) extends Matcher[Vector[4]]:

    def apply(left: Vector[4]): MatchResult =
      MatchResult(
        left.dst(expected) < Const.epsilon,
        s"""${left.toString} is not epsilon-equal to ${expected.toString}""",
        s"""${left.toString} equal to ${expected.toString} to ${Const.epsilon}"""
      )

  def epsilonEqual(expected: Vector[4]) = new VectorsRoughlyEqualMatcher(expected)

  class MatricesRoughlyEqualMatcher(expected: Matrix[4]) extends Matcher[Matrix[4]]:

    def apply(left: Matrix[4]): MatchResult = {
      MatchResult(
        expected.m.zip(left.m).forall(
          (expectedElement, leftElement) => expectedElement - leftElement < Const.epsilon
        ),
        s"""$left is not epsilon-equal to \n$expected""",
        s"""$left equal to $expected to ${Const.epsilon}"""
      )
    }

  def epsilonEqual(expected: Matrix[4]) = new MatricesRoughlyEqualMatcher(expected)

  def containsEpsilon(left: Iterable[Vector[4]], expected: Vector[4]): Boolean =
    left.exists(_.dst(expected) < Const.epsilon)

  class ContainerContainsVector4Matcher(expected: Vector[4]) extends Matcher[Iterable[Vector[4]]]:
    def apply(left: Iterable[Vector[4]]): MatchResult =
      MatchResult(
        containsEpsilon(left, expected),
        s"""${left.map(_.toString)} does not contain ${expected.toString}""",
        s"""${left.map(_.toString)} contains ${expected.toString} to ${Const.epsilon}"""
      )

  def containEpsilon(expected: Vector[4]) = new ContainerContainsVector4Matcher(expected)

  class FaceContainsAllVector4Matcher(expected: Iterable[Vector[4]]) extends Matcher[Seq[Face4D]]:
    def apply(left: Seq[Face4D]): MatchResult =
      MatchResult(
        containsAllEpsilon(left, expected.toSeq),
        s"""${left.map(_.toString)} does not contain all ${expected.map(_.toString)}""",
        s"""${left.map(_.toString)} contains all ${expected.map(_.toString)}"""
      )

  def containAllEpsilon(expected: Iterable[Vector[4]]) = new FaceContainsAllVector4Matcher(expected)

  private def containsAllEpsilon(rects: Seq[Face4D], expected: Seq[Vector[4]]): Boolean =
    rects.map(_.asSeq).exists(rect => rect.forall(v => containsEpsilon(expected, v)))

  class Vector4SeqsMatcher(expected: Iterable[Vector[4]]) extends Matcher[Iterable[Vector[4]]]:
    def apply(left: Iterable[Vector[4]]): MatchResult =
      MatchResult(
        left.zip(expected).forall((v1, v2) => v1 === v2),
        s"""${left.map(_.toString)} not equal ${expected.map(_.toString)}""",
        s"""${left.map(_.toString)} equals ${expected.map(_.toString)} to ${Const.epsilon}"""
      )

  def epsilonEqual(expected: Iterable[Vector[4]]) = new Vector4SeqsMatcher(expected)


object CustomMatchers extends CustomMatchers

