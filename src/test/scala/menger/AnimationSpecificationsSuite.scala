package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AnimationSpecificationsSuite extends AnyFlatSpec with Matchers:
  "AnimationSpecifications" should "be empty by default" in:
    AnimationSpecifications().parts should be (empty)

  "Finding the animation specification for a frame" should "return the frame if only one part" in:
    AnimationSpecifications(List("frames=10:rot-x-w=0-10")).partAndFrame(5) should be (0, 5)