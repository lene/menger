package menger

import com.badlogic.gdx.Version
import org.scalatest.flatspec.AnyFlatSpec

class BasicsSuite  extends AnyFlatSpec:
  "libGDX version" should "be high enough" in:
    assert(Version.isHigherEqual(1, 12, 0))
