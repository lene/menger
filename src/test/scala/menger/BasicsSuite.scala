package menger

import com.badlogic.gdx.Version
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BasicsSuite  extends AnyFlatSpec with Matchers:
  "libGDX version" should "be high enough" in:
    Version.isHigherEqual(1, 12, 0) should be (true)
