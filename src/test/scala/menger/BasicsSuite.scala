package menger

import com.badlogic.gdx.Version
import org.scalatest.funsuite.AnyFunSuite

class BasicsSuite  extends AnyFunSuite:
  test("libGDX version is high enough") {
    assert(Version.isHigherEqual(1, 12, 0))
  }
