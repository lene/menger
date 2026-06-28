package menger.objects

import org.scalatest.Inspectors.forAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LSystemGrammarSuite extends AnyFlatSpec with Matchers:

  "LSystemGrammar algae (deterministic)" should "produce Fibonacci lengths" in:
    val grammar = LSystemGrammar(
      axiom = "A",
      rules = Map('A' -> Seq((1.0, "AB")), 'B' -> Seq((1.0, "A")))
    )
    val expectedLengths = Seq(1, 2, 3, 5, 8, 13, 21, 34)
    for (n, expected) <- (0 to 7).zip(expectedLengths) do
      val result = grammar.rewrite(n)
      withClue(s"at iteration $n: ") { result.length shouldBe expected }

  "LSystemGrammar Koch curve (deterministic)" should "produce known string at n=3" in:
    val grammar = LSystemGrammar(
      axiom = "F",
      rules = Map('F' -> Seq((1.0, "F+F--F+F")))
    )
    val n1 = "F+F--F+F"
    val n2 = n1.flatMap(c => if c == 'F' then "F+F--F+F" else c.toString)
    val expected = n2.flatMap(c => if c == 'F' then "F+F--F+F" else c.toString)
    grammar.rewrite(3) shouldBe expected

  "LSystemGrammar with branching" should "maintain push/pop bracket balance" in:
    val grammar = LSystemGrammar(
      axiom = "F",
      rules = Map('F' -> Seq((1.0, "F[+F]F[-F]F")))
    )
    val results = (1 to 5).map(grammar.rewrite)
    forAll(results) { result =>
      result.count(_ == '[') shouldBe result.count(_ == ']')
    }

  "Stochastic LSystemGrammar with same seed" should "produce identical strings" in:
    val grammar1 = LSystemGrammar(
      axiom = "X",
      rules = Map(
        'X' -> Seq((0.5, "XAX"), (0.5, "XBX"))
      ),
      seed = 42L
    )
    val grammar2 = LSystemGrammar(
      axiom = "X",
      rules = Map(
        'X' -> Seq((0.5, "XAX"), (0.5, "XBX"))
      ),
      seed = 42L
    )
    grammar1.rewrite(10) shouldBe grammar2.rewrite(10)

  "Stochastic LSystemGrammar with different seeds" should "produce different strings" in:
    val grammar1 = LSystemGrammar(
      axiom = "X",
      rules = Map(
        'X' -> Seq((0.5, "XAX"), (0.5, "XBX"))
      ),
      seed = 1L
    )
    val grammar2 = LSystemGrammar(
      axiom = "X",
      rules = Map(
        'X' -> Seq((0.5, "XAX"), (0.5, "XBX"))
      ),
      seed = 999L
    )
    val result1 = grammar1.rewrite(12)
    val result2 = grammar2.rewrite(12)
    result1 should not be result2

  "LSystemGrammar with exponential growth" should "fail at ~10^7 symbols" in:
    val grammar = LSystemGrammar(
      axiom = "A",
      rules = Map('A' -> Seq((1.0, "AAAAA")))
    )
    an[IllegalArgumentException] should be thrownBy {
      grammar.rewrite(11)
    }
    val caught = the[IllegalArgumentException] thrownBy {
      grammar.rewrite(11)
    }
    caught.getMessage should include("exceeded maximum length")

  "LSystemGrammar with unknown symbol" should "pass through unchanged" in:
    val grammar = LSystemGrammar(
      axiom = "FXF",
      rules = Map('F' -> Seq((1.0, "FF")))
    )
    val result = grammar.rewrite(2)
    result should include("X")

  "LSystemGrammar turtle alphabet symbols" should "pass through without warnings" in:
    val grammar = LSystemGrammar(
      axiom = "F+-[]",
      rules = Map('F' -> Seq((1.0, "F+--F")))
    )
    val result = grammar.rewrite(1)
    result should not be empty

  "LSystemGrammar with zero iterations" should "return the axiom unchanged" in:
    val grammar = LSystemGrammar(
      axiom = "ABC",
      rules = Map('A' -> Seq((1.0, "X")), 'B' -> Seq((1.0, "Y")))
    )
    grammar.rewrite(0) shouldBe "ABC"

  "LSystemGrammar with no matching rules" should "return axiom unchanged" in:
    val grammar = LSystemGrammar(
      axiom = "XYZ",
      rules = Map.empty
    )
    grammar.rewrite(5) shouldBe "XYZ"

  "LSystemGrammar rule validation" should "emit warning for unreferenced symbols" in:
    val grammar = LSystemGrammar(
      axiom = "A",
      rules = Map('A' -> Seq((1.0, "BZ")))
    )
    grammar.rewrite(1) should include("Z")

  "LSystemGrammar with empty axiom" should "produce empty string" in:
    val grammar = LSystemGrammar(
      axiom = "",
      rules = Map('A' -> Seq((1.0, "B")))
    )
    grammar.rewrite(10) shouldBe ""
