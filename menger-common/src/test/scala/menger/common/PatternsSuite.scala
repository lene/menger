package menger.common

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PatternsSuite extends AnyFlatSpec with Matchers:

  "CompositeType pattern" should "match valid composite specifications" in:
    "composite[cube,square]" match
      case Patterns.CompositeType(content) => content shouldBe "cube,square"
      case _ => fail("Should match composite pattern")

  it should "capture single component" in:
    "composite[cube]" match
      case Patterns.CompositeType(content) => content shouldBe "cube"
      case _ => fail("Should match composite pattern")

  it should "not match non-composite strings" in:
    "cube" match
      case Patterns.CompositeType(_) => fail("Should not match non-composite string")
      case _ => succeed

  it should "not match malformed composite strings" in:
    "composite[cube" match
      case Patterns.CompositeType(_) => fail("Should not match unclosed bracket")
      case _ => succeed

  it should "not match empty composite" in:
    "composite[]" match
      case Patterns.CompositeType(content) => fail(s"Should not match empty composite, got: $content")
      case _ => succeed

  "parseCompositeComponents" should "split comma-separated components" in:
    Patterns.parseCompositeComponents("cube,square") shouldBe List("cube", "square")

  it should "handle single component" in:
    Patterns.parseCompositeComponents("cube") shouldBe List("cube")

  it should "handle multiple components" in:
    Patterns.parseCompositeComponents("a,b,c,d") shouldBe List("a", "b", "c", "d")

  it should "preserve whitespace in components" in:
    Patterns.parseCompositeComponents("a, b") shouldBe List("a", " b")
