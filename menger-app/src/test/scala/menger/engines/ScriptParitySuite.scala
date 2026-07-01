package menger.engines

import menger.common.ObjectType
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source
import scala.util.Using

/** Script-parity fitness function (T9, Sprint 32).
  *
  * Extracts type=<...> tokens from integration-tests.sh and manual-test.sh,
  * asserts integration ⊇ manual coverage, and verifies every VALID_TYPES entry
  * appears in at least one script. Fails CI on divergence.
  */
class ScriptParitySuite extends AnyFlatSpec with Matchers:

  private val scriptsRoot: String =
    val dirs = List("scripts", "../scripts", "../../scripts", "../../../scripts",
      "menger-app/../scripts", "/root/projects/menger/scripts")
    dirs.find(d => java.io.File(d, "integration-tests.sh").exists())
      .getOrElse(sys.error("Cannot find scripts directory"))

  private def extractTypes(scriptName: String): Set[String] =
    val path = s"$scriptsRoot/$scriptName"
    Using.resource(Source.fromFile(path)): source =>
      val typeRegex = """type=([a-z0-9-]+)""".r
      source.getLines
        .flatMap(typeRegex.findAllMatchIn(_))
        .map(m => ObjectType.normalize(m.group(1)))
        .toSet

  private val integrationTypes: Set[String] = extractTypes("integration-tests.sh")
  private val manualTypes: Set[String] = extractTypes("manual-test.sh")
  private val coveredTypes: Set[String] = integrationTypes ++ manualTypes

  // Types that are DSL-only and cannot appear as type=<type> in CLI test scripts
  private val dslOnlyTypes: Set[String] = Set("parametric")

  "Script-parity fitness function" should "cover all manual-test types in integration-tests" in:
    val missingInIntegration = (manualTypes -- integrationTypes) -- dslOnlyTypes
    withClue(s"Types in manual-test.sh but not integration-tests.sh: ${missingInIntegration.mkString(", ")}"):
      missingInIntegration shouldBe empty

  it should "have every VALID_TYPES entry in at least one script" in:
    val uncovered = ObjectType.VALID_TYPES.diff(coveredTypes).diff(dslOnlyTypes)
    withClue(s"VALID_TYPES not covered by any test script: ${uncovered.mkString(", ")}"):
      uncovered shouldBe empty

  it should "not have empty type extraction from either script" in:
    integrationTypes should not be empty
    manualTypes should not be empty
