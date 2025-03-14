package menger.objects.higher_d

import com.badlogic.gdx.math.Vector4
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
        s"""${left.map(asString).mkString(", ")} does not contain ${expected.asString}""",
        s"""${left.map(asString).mkString(", ")} contains ${expected.asString} to ${Const.epsilon}"""
      )

  def containEpsilon(expected: Vector4) = new ContainerContainsVector4Matcher(expected)

  class FaceContainsAllVector4Matcher(expected: Iterable[Vector4]) extends Matcher[Seq[Face4D]]:
    def apply(left: Seq[Face4D]): MatchResult =
      MatchResult(
        containsAllEpsilon(left, expected.toSeq),
        s"""${left.map(_.toString).mkString(", ")} does not contain all ${expected.map(asString).mkString(", ")}""",
        s"""${left.map(_.toString).mkString(", ")} contains all ${expected.map(asString).mkString(", ")}"""
      )

  def containAllEpsilon(expected: Iterable[Vector4]) = new FaceContainsAllVector4Matcher(expected)

  /**
   * Separate class for negating the containAllEpsilon matcher since using it with `not` gives the error:
   * value containAllEpsilon is not a member of
   * org.scalatest.matchers.dsl.ResultOfNotWordForAny[Seq[menger.objects.higher_d.Face4D]]
   */
  class FaceNotContainsAllVector4Matcher(expected: Iterable[Vector4]) extends Matcher[Seq[Face4D]]:
    def apply(left: Seq[Face4D]): MatchResult =
      MatchResult(
        !containsAllEpsilon(left, expected.toSeq),
        s"""${left.map(_.toString).mkString(", ")} contains all ${expected.map(asString).mkString(", ")}""",
        s"""${left.map(_.toString).mkString(", ")} does not contain all ${expected.map(asString).mkString(", ")}"""
      )

  def notContainAllEpsilon(expected: Iterable[Vector4]) = new FaceNotContainsAllVector4Matcher(expected)

  private def containsAllEpsilon(rects: Seq[Face4D], expected: Seq[Vector4]): Boolean =
    rects.map(_.asSeq).exists(rect => rect.forall(v => containsEpsilon(expected, v)))


object CustomMatchers extends CustomMatchers

