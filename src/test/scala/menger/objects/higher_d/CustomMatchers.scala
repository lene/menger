package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
import menger.Const
import org.scalatest.matchers.{MatchResult, Matcher}

trait CustomMatchers:
  class VectorsRoughlyEqualMatcher(expected: Vector4) extends Matcher[Vector4]:

    def apply(left: Vector4): MatchResult =
      MatchResult(
        left.dst(expected) < Const.epsilon,
        s"""${left.asString} is not equal to ${expected.asString}""",
        s"""${left.asString} equal to ${expected.asString} to ${Const.epsilon}"""
      )

  def epsilonEqual(expected: Vector4) = new VectorsRoughlyEqualMatcher(expected)

  def containsEpsilon(left: Iterable[Vector4], expected: Vector4): Boolean =
    left.exists(_.dst(expected) < Const.epsilon)

  class ContainerContainsVector4Matcher(expected: Vector4) extends Matcher[Iterable[Vector4]]:
    def apply(left: Iterable[Vector4]): MatchResult =
      MatchResult(
        containsEpsilon(left, expected),
        s"""${left.map(asString)} does not contain ${expected.asString}""",
        s"""${left.map(asString)} contains ${expected.asString} to ${Const.epsilon}"""
      )

  def containEpsilon(expected: Vector4) = new ContainerContainsVector4Matcher(expected)

  class FaceContainsAllVector4Matcher(expected: Iterable[Vector4]) extends Matcher[Seq[Face4D]]:
    def apply(left: Seq[Face4D]): MatchResult =
      MatchResult(
        containsAllEpsilon(left, expected.toSeq),
        s"""${left.map(_.toString)} does not contain all ${expected.map(asString)}""",
        s"""${left.map(_.toString)} contains all ${expected.map(asString)}"""
      )

  def containAllEpsilon(expected: Iterable[Vector4]) = new FaceContainsAllVector4Matcher(expected)

  /**
   * Separate class for negating the containAllEpsilon matcher since using it with `not` gives the error:
   * value containAllEpsilon is not a member of
   * org.scalatest.matchers.dsl.ResultOfNotWordForAny[Seq[Face4D]]
   */
  class FaceNotContainsAllVector4Matcher(expected: Iterable[Vector4]) extends Matcher[Seq[Face4D]]:
    def apply(left: Seq[Face4D]): MatchResult =
      MatchResult(
        !containsAllEpsilon(left, expected.toSeq),
        s"""${left.map(_.toString)} contains all ${expected.map(asString)}""",
        s"""${left.map(_.toString)} does not contain all ${expected.map(asString)}"""
      )

  def notContainAllEpsilon(expected: Iterable[Vector4]) = new FaceNotContainsAllVector4Matcher(expected)

  private def containsAllEpsilon(rects: Seq[Face4D], expected: Seq[Vector4]): Boolean =
    rects.map(_.asSeq).exists(rect => rect.forall(v => containsEpsilon(expected, v)))

  class Vector4SeqsMatcher(expected: Iterable[Vector4]) extends Matcher[Iterable[Vector4]]:
    def apply(left: Iterable[Vector4]): MatchResult =
      MatchResult(
        left.zip(expected).forall((v1, v2) => v1.epsilonEquals(v2)),
        s"""${left.map(asString)} not equal ${expected.map(asString)}""",
        s"""${left.map(asString)} equals ${expected.map(asString)} to ${Const.epsilon}"""
      )

  def epsilonEqual(expected: Iterable[Vector4]) = new Vector4SeqsMatcher(expected)


object CustomMatchers extends CustomMatchers

