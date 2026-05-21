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

  // Blocked by P0.A: menger.dsl.SceneConverter imports menger.config.{PlaneConfig,CameraConfig,SceneConfig}
  // and menger.optix.{CausticsConfig,RenderConfig}. Fix: move SceneConverter to menger.engines.
  // Also blocked by P0.B: menger.dsl.Material imports menger.optix.Material for preset delegation.
  // Unblocking requires moving all dsl→optix conversions to the engines layer.
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

  // Fixed: removed VideoEncoder.checkAvailable call from CliValidation (checked at VideoEngine init)
  "menger.cli" should "not depend on engines or optix in menger.cli" in:
    noClasses().that().resideInAPackage("menger.cli..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("menger.engines..", "menger.optix..")
      .check(allClasses)

  // Blocked: Scala `val` fields in case classes compile to non-final JVM fields,
  // so haveOnlyFinalFields() fires on all case classes even with no `var` fields.
  // Fix: rewrite the rule using a custom DescribedPredicate that checks Scala-level
  // mutability (inspect @scala.annotation.varargs or use ArchUnit field.isFinal + filter
  // companion modules), or wait for ArchUnit native Scala support.
  ignore should "have only final fields in menger.common" in:
    noClasses().that().resideInAPackage("menger.common..")
      .should().haveOnlyFinalFields()
      .check(allClasses)

  // Blocked: menger.dsl.SceneRegistry uses scala.collection.mutable.Map for the scene registry.
  // Fix: replace with an immutable approach (AtomicReference[Map[String,Scene]] or move the
  // registry to a higher layer). Independent of P0.A — can be fixed without moving SceneConverter.
  ignore should "not use mutable collections in menger.dsl" in:
    noClasses().that().resideInAPackage("menger.dsl..")
      .should().dependOnClassesThat()
        .resideInAPackage("scala.collection.mutable..")
      .check(allClasses)

  // Blocked: Scala case classes and companions implement java.io.Serializable implicitly at
  // the bytecode level, so resideInAnyPackage("java.io..") fires even though no source file
  // in menger.common has an explicit java.io import. Fix: rewrite the rule to exclude
  // java.io.Serializable specifically using a custom DescribedPredicate, e.g.:
  //   dependOnClassesThat(not(name("java.io.Serializable")).and(resideInAnyPackage("java.io..")))
  ignore should "not use file IO in menger.common" in:
    noClasses().that().resideInAPackage("menger.common..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("java.io..", "java.nio.file..")
      .check(allClasses)

  // Blocked: 4 classes use slf4j via LazyLogging:
  //   - ParametricTessellator: logs a memory warning for very large tessellations (justified)
  //   - higher_d/Rotation: one debug log for rotation computation
  //   - higher_d/Plane: imports LazyLogging but never calls logger (dead import)
  //   - higher_d/TesseractSponge2: imports LazyLogging but never calls logger (dead import)
  // Fix: remove dead LazyLogging from Plane and TesseractSponge2; move ParametricTessellator
  // warning to caller; replace Rotation debug with a comment or remove it.
  // Note: java.io.Serializable IS a problem — Scala case classes implement it implicitly at
  // bytecode level, so resideInAnyPackage("java.io..") fires. Rule must be rewritten to
  // exclude java.io.Serializable (same fix as the menger.common file IO rule above).
  ignore should "not use file IO or logging in menger.objects" in:
    noClasses().that().resideInAPackage("menger.objects..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("java.io..", "java.nio.file..", "org.slf4j..")
      .check(allClasses)

  // Rule 2.4 (sealed hierarchies) is not expressible in ArchUnit — enforced by code review.
  // dsl scene-node traits and MengerException subtypes must be sealed.
