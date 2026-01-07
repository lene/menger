package menger.common

import scala.util.matching.Regex

object Patterns:

  /**
   * Regex pattern for parsing composite type specifications.
   *
   * Format: "composite[type1,type2,...]"
   * - composite\[: literal string "composite["
   * - (.+): one or more characters (captured group 1 - comma-separated types)
   * - ]: literal closing bracket
   *
   * Example: "composite[cube,square]" matches with capture group 1 = "cube,square"
   */
  val CompositeType: Regex = """composite\[(.+)]""".r

  /**
   * Extracts component type names from a composite specification string.
   *
   * @param compositeContent The captured content from composite pattern (e.g., "cube,square")
   * @return List of individual component type names (e.g., List("cube", "square"))
   */
  def parseCompositeComponents(compositeContent: String): List[String] =
    compositeContent.split(",").toList
