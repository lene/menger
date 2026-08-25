package menger.engines

import menger.common.ObjectType
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source
import scala.util.Using

/** Script-parity fitness function (T9, Sprint 32; extended F14, Sprint 35 Ph4).
  *
  * Extracts feature tokens from integration-tests.sh and manual-test.sh, asserts
  * bidirectional coverage on materials and flags, manual ⊆ integration on types
  * and DSL scenes, and verifies every VALID_TYPES entry appears in at least one
  * script. The required coverage set is declared in scripts/coverage-manifest.yaml.
  */
class ScriptParitySuite extends AnyFlatSpec with Matchers:

  private val scriptsRoot: String =
    val dirs = List("scripts", "../scripts", "../../scripts", "../../../scripts",
      "menger-app/../scripts", "/root/projects/menger/scripts")
    dirs.find(d => java.io.File(d, "integration-tests.sh").exists())
      .getOrElse(sys.error("Cannot find scripts directory"))

  private def readScript(scriptName: String): String =
    val path = s"$scriptsRoot/$scriptName"
    Using.resource(Source.fromFile(path))(_.mkString)

  private def extractTypes(scriptName: String): Set[String] =
    val typeRegex = """type=([a-z0-9-]+)""".r
    typeRegex.findAllMatchIn(readScript(scriptName))
      .map(m => ObjectType.normalize(m.group(1))).toSet

  private def extractPattern(scriptName: String, pattern: String): Set[String] =
    val regex = pattern.r
    regex.findAllMatchIn(readScript(scriptName)).map(_.group(1)).toSet

  private def scriptContains(scriptName: String, token: String): Boolean =
    readScript(scriptName).contains(token)

  // ── coverage manifest ────────────────────────────────────────────────────

  private val manifestPath = s"$scriptsRoot/coverage-manifest.yaml"
  private val manifestText = Using.resource(Source.fromFile(manifestPath))(_.mkString)

  private def manifestSection(name: String): List[String] =
    val sectionRegex = s"""(?m)^$name:$$([\\s\\S]*?)(?=^\\S+:|\\Z)""".r
    sectionRegex.findFirstMatchIn(manifestText) match
      case None => Nil
      case Some(m) =>
        val itemRegex = """(?m)^\s+-\s+(\S+)""".r
        itemRegex.findAllMatchIn(m.group(1)).map(_.group(1)).toList

  private val requiredMaterials: Set[String] = manifestSection("materials").toSet
  private val requiredFlags: Set[String] = manifestSection("rendering_flags").toSet
  private val requiredFieldCombinations: List[(String, String)] =
    manifestSection("field_combinations").map { pair =>
      val fields = pair.split("\\+")
      (fields(0), fields(1))
    }

  // ── extracted tokens ─────────────────────────────────────────────────────

  private val integrationTypes: Set[String] = extractTypes("integration-tests.sh")
  private val manualTypes: Set[String] = extractTypes("manual-test.sh")
  private val coveredTypes: Set[String] = integrationTypes ++ manualTypes

  private val integrationMaterials: Set[String] =
    extractPattern("integration-tests.sh", """material=([a-z][-a-z]*)""")
  private val manualMaterials: Set[String] =
    extractPattern("manual-test.sh", """material=([a-z][-a-z]*)""")

  private def objectsClauses(scriptName: String): List[String] =
    val regex = """--objects\s+"?([^"\n]+)"?""".r
    regex.findAllMatchIn(readScript(scriptName)).map(_.group(1)).toList

  private val integrationObjectsClauses: List[String] = objectsClauses("integration-tests.sh")
  private val manualObjectsClauses: List[String] = objectsClauses("manual-test.sh")

  private val integrationScenes: Set[String] =
    extractPattern("integration-tests.sh", """--scene (examples\.dsl\.\w+)""")
  private val manualScenes: Set[String] =
    extractPattern("manual-test.sh", """--scene (examples\.dsl\.\w+)""")

  // Types that are DSL-only and cannot appear as type=<type> in CLI test scripts
  private val dslOnlyTypes: Set[String] = Set("parametric")
  // Materials used only in negative tests (deliberately invalid)
  private val negativeTestMaterials: Set[String] = Set("unobtanium")

  // ── type parity (unchanged, T9) ──────────────────────────────────────────

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

  // ── material parity (F14, Sprint 35 Ph4) ─────────────────────────────────

  it should "exercise every manifest material in both scripts" in:
    requiredMaterials.foreach: mat =>
      withClue(s"material=$mat missing from integration-tests.sh: "):
        integrationMaterials should contain (mat)
      withClue(s"material=$mat missing from manual-test.sh: "):
        manualMaterials should contain (mat)

  it should "not have unregistered materials (excluding negative tests)" in:
    val unregistered = (integrationMaterials ++ manualMaterials) --
      requiredMaterials -- negativeTestMaterials
    withClue(s"Materials in scripts but not in coverage-manifest.yaml: ${unregistered.mkString(", ")}"):
      unregistered shouldBe empty

  // ── flag parity (F14, Sprint 35 Ph4) ─────────────────────────────────────

  it should "exercise every manifest rendering flag in both scripts" in:
    requiredFlags.foreach: flag =>
      withClue(s"$flag missing from integration-tests.sh: "):
        scriptContains("integration-tests.sh", flag) shouldBe true
      withClue(s"$flag missing from manual-test.sh: "):
        scriptContains("manual-test.sh", flag) shouldBe true

  // ── DSL scene coverage (F14, Sprint 35 Ph4) ──────────────────────────────

  it should "cover all manual DSL scenes in integration-tests" in:
    val missingInIntegration = manualScenes -- integrationScenes
    withClue(s"DSL scenes in manual-test.sh but not integration-tests.sh: ${missingInIntegration.mkString(", ")}"):
      missingInIntegration shouldBe empty

  // ── field-combination coverage (Sprint 36 #18, QA_INCIDENTS.md 2026-08-21) ──
  // Presence checks above ("material=X appears somewhere") don't catch a field
  // silently ignored whenever a specific OTHER field is also set. A declared pair
  // must appear together (both `field=` tokens) in the same --objects clause.

  it should "exercise every declared field combination together in both scripts" in:
    requiredFieldCombinations.foreach: (fieldA, fieldB) =>
      def coveredBy(clauses: List[String]): Boolean =
        clauses.exists(c => c.contains(s"$fieldA=") && c.contains(s"$fieldB="))
      withClue(s"No --objects clause in integration-tests.sh sets both $fieldA= and $fieldB=: "):
        coveredBy(integrationObjectsClauses) shouldBe true
      withClue(s"No --objects clause in manual-test.sh sets both $fieldA= and $fieldB=: "):
        coveredBy(manualObjectsClauses) shouldBe true
