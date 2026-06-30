package menger.objects

import scala.util.Random

case class LSystemGrammar(
  axiom: String,
  rules: Map[Char, Seq[(Double, String)]],
  seed: Long = 42L
):
  private val turtleAlphabet: Set[Char] = "Ff+-&^\\\\/|[]!'%()><".toSet
  private val maxLength = 10_000_000
  private val rng = Random(seed)

  def rewrite(iterations: Int): String =
    require(iterations >= 0, s"iterations must be non-negative, got $iterations")
    (0 until iterations).foldLeft(axiom)((current, _) => rewriteOnce(current))

  private def rewriteOnce(input: String): String =
    val result = input.flatMap { ch =>
      rules.get(ch) match
        case Some(productions) => selectProduction(productions)
        case None => ch.toString
    }
    require(
      result.length <= maxLength,
      s"L-system string exceeded maximum length of $maxLength " +
        s"(got ${result.length}). Reduce iteration count."
    )
    result

  private def selectProduction(productions: Seq[(Double, String)]): String =
    productions match
      case Seq((_, singleRule)) => singleRule
      case weighted =>
        val totalWeight = weighted.map(_._1).sum
        val roll = rng.nextDouble() * totalWeight
        weighted
          .scanLeft((0.0, "")) { case ((cum, _), (w, prod)) => (cum + w, prod) }
          .drop(1)
          .find { case (cum, _) => roll < cum }
          .map(_._2)
          .getOrElse(weighted.last._2)
