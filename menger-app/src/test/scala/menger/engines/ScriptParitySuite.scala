package menger.engines

import menger.cli.MengerCLIOptions
import menger.spec.ObjectType
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

  /** The flag itself, not a longer flag it prefixes (`--rot` is not covered by `--rot-x`). */
  private def scriptHasFlag(scriptName: String, flag: String): Boolean =
    (java.util.regex.Pattern.quote(flag) + "(?![a-z0-9-])").r.findFirstIn(readScript(scriptName)).isDefined

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
        scriptHasFlag("integration-tests.sh", flag) shouldBe true
      withClue(s"$flag missing from manual-test.sh: "):
        scriptHasFlag("manual-test.sh", flag) shouldBe true

  // Review 2026-10-14 (menger#93): the required flags were a hand-picked list (21 of 63), so a
  // new flag -- including agent-facing ones like --preview, --playback, --display -- never had
  // to be covered. Every flag MengerCLIOptions defines is now either required in both scripts
  // (rendering_flags), exempt with a reason (flags_exempt), or on the named backlog of flags
  // that still need a script entry (flags_backlog); a flag in none of them fails.
  private val allFlags: Set[String] =
    MengerCLIOptions(Seq.empty).builder.opts.map(o => s"--${o.name}").toSet
  private val exemptFlags: Set[String] = manifestSection("flags_exempt").toSet
  private val backlogFlags: Set[String] = manifestSection("flags_backlog").toSet

  it should "classify every CLI flag as required, exempt or backlog" in:
    val unclassified = allFlags -- requiredFlags -- exemptFlags -- backlogFlags
    withClue(s"CLI flags in no coverage-manifest.yaml list: ${unclassified.toList.sorted.mkString(" ")}"):
      unclassified shouldBe empty

  it should "list only real CLI flags in coverage-manifest.yaml" in:
    val stale = (requiredFlags ++ exemptFlags ++ backlogFlags) -- allFlags
    withClue(s"coverage-manifest.yaml names flags MengerCLIOptions does not define: ${stale.mkString(" ")}"):
      stale shouldBe empty

  it should "not keep a flag on the backlog once both scripts cover it" in:
    val covered = backlogFlags.filter(f =>
      scriptHasFlag("integration-tests.sh", f) && scriptHasFlag("manual-test.sh", f))
    withClue(s"move from flags_backlog to rendering_flags: ${covered.mkString(" ")}"):
      covered shouldBe empty

  // ── DSL scene coverage (F14, Sprint 35 Ph4) ──────────────────────────────

  // menger#93: every example scene (the corpus the agent learns from) is in the integration
  // suite or on the named scenes_backlog; a new scene in neither fails.
  private val exampleScenes: Set[String] =
    val dir = java.io.File(s"$scriptsRoot/../menger-app/src/main/scala/examples/dsl")
    Option(dir.listFiles()).toList.flatten
      .map(_.getName).filter(n => n.endsWith(".scala") && n != "SceneIndex.scala")
      .map(n => s"examples.dsl.${n.stripSuffix(".scala")}").toSet
  private val sceneBacklog: Set[String] =
    manifestSection("scenes_backlog").map(n => s"examples.dsl.$n").toSet

  it should "cover every example scene in integration-tests or list it on scenes_backlog" in:
    exampleScenes should not be empty
    val uncovered = exampleScenes -- integrationScenes -- sceneBacklog
    withClue(s"example scenes in neither integration-tests.sh nor scenes_backlog: ${uncovered.toList.sorted.mkString(" ")}"):
      uncovered shouldBe empty
    val done = sceneBacklog intersect integrationScenes
    withClue(s"remove from scenes_backlog, integration-tests.sh covers them: ${done.mkString(" ")}"):
      done shouldBe empty

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
