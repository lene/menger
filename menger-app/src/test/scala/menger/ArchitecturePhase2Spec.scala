package menger

import com.tngtech.archunit.core.importer.ClassFileImporter
import com.tngtech.archunit.core.importer.ImportOption
import com.tngtech.archunit.core.importer.ImportOption.DoNotIncludeTests
import com.tngtech.archunit.core.importer.Location
import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Phase 2 architectural rules.
 * P0.B (objects→input decoupling) is resolved; per-test ignore marks remaining blockers.
 * Remove ignore from each test once the corresponding blocker is resolved.
 */
class ArchitecturePhase2Spec extends AnyFlatSpec with Matchers:

  private val doNotIncludeSbtTestClasses: ImportOption =
    (location: Location) => !location.matches(java.util.regex.Pattern.compile(".*test-classes.*"))

  private lazy val allClasses =
    ClassFileImporter()
      .withImportOption(DoNotIncludeTests())
      .withImportOption(doNotIncludeSbtTestClasses)
      .importPackages("menger")

  // P0.B resolved: menger.objects no longer imports menger.input
  "menger.objects" should "depend only on menger.common (P0.B resolved)" in:
    noClasses().that().resideInAPackage("menger.objects..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("menger.input..", "menger.engines..",
                            "menger.dsl..", "menger.cli..", "menger.config..")
      .check(allClasses)

  // Blocked by P0.A: menger.dsl still imports menger.config
  ignore should "depend only on menger.common and menger.objects (P0.A blocker)" in:
    noClasses().that().resideInAPackage("menger.dsl..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("menger.engines..", "menger.cli..",
                            "menger.config..", "menger.input..")
      .check(allClasses)

  "menger.config" should "not depend on engines or cli" in:
    noClasses().that().resideInAPackage("menger.config..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("menger.engines..", "menger.cli..")
      .check(allClasses)

  "menger.input" should "not depend on engines, dsl, or config" in:
    noClasses().that().resideInAPackage("menger.input..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("menger.engines..", "menger.dsl..", "menger.config..")
      .check(allClasses)

  // Blocked: menger.cli.CliValidation calls menger.engines.VideoEncoder.checkAvailable
  ignore should "not depend on engines or optix in menger.cli" in:
    noClasses().that().resideInAPackage("menger.cli..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("menger.engines..", "menger.optix..")
      .check(allClasses)

  // Immutability — audit required before un-ignoring
  ignore should "have only final fields in menger.common" in:
    noClasses().that().resideInAPackage("menger.common..")
      .should().haveOnlyFinalFields()
      .check(allClasses)

  // Blocked by P0.A: dsl→config violation and mutable usage not yet resolved
  ignore should "not use mutable collections in menger.dsl (P0.A blocker)" in:
    noClasses().that().resideInAPackage("menger.dsl..")
      .should().dependOnClassesThat()
        .resideInAPackage("scala.collection.mutable..")
      .check(allClasses)

  // Purity — audit required before un-ignoring
  ignore should "not use file IO in menger.common" in:
    noClasses().that().resideInAPackage("menger.common..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("java.io..", "java.nio.file..")
      .check(allClasses)

  // Blocked: Scala case classes implement java.io.Serializable implicitly; ParametricTessellator and Rotation use slf4j
  ignore should "not use file IO or logging in menger.objects" in:
    noClasses().that().resideInAPackage("menger.objects..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("java.io..", "java.nio.file..", "org.slf4j..")
      .check(allClasses)

  // Rule 2.4 (sealed hierarchies) is not expressible in ArchUnit — enforced by code review.
  // dsl scene-node traits and MengerException subtypes must be sealed.
