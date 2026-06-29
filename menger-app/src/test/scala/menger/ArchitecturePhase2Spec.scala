package menger

import com.tngtech.archunit.base.DescribedPredicate
import com.tngtech.archunit.core.domain.JavaClass
import com.tngtech.archunit.core.importer.ClassFileImporter
import com.tngtech.archunit.core.importer.ImportOption
import com.tngtech.archunit.core.importer.ImportOption.DoNotIncludeTests
import com.tngtech.archunit.core.importer.Location
import com.tngtech.archunit.lang.ArchCondition
import com.tngtech.archunit.lang.ConditionEvents
import com.tngtech.archunit.lang.SimpleConditionEvent
import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Phase 2 architectural rules.
 */
class ArchitecturePhase2Spec extends AnyFlatSpec with Matchers:

  private val doNotIncludeSbtTestClasses: ImportOption =
    (location: Location) => !location.matches(java.util.regex.Pattern.compile(".*test-classes.*"))

  private lazy val allClasses =
    ClassFileImporter()
      .withImportOption(DoNotIncludeTests())
      .withImportOption(doNotIncludeSbtTestClasses)
      .importPackages("menger")

  // Scala `val` fields compile to non-final JVM fields, so haveOnlyFinalFields() fires even
  // with no `var` fields. This condition checks for var-generated setter methods instead:
  // Scala compiles `var foo` into a getter `foo()` and setter `foo_$eq(x)`. A class with no
  // methods ending in `_$eq` has no mutable fields at the Scala source level.
  private val haveNoVarFields: ArchCondition[JavaClass] =
    new ArchCondition[JavaClass]("have var fields (Scala setter methods)"):
      override def check(clazz: JavaClass, events: ConditionEvents): Unit =
        val setters = clazz.getMethods.stream()
          .filter(m => m.getName.endsWith("_$eq"))
          .toList
        if !setters.isEmpty then
          events.add(SimpleConditionEvent.violated(
            clazz,
            s"${clazz.getName} has var-generated setter(s): " +
              setters.stream().map(_.getName).toList.toString
          ))

  // Scala case classes implement java.io.Serializable implicitly at the bytecode level,
  // so checking for java.io dependencies fires even without explicit java.io imports.
  // This predicate matches java.io classes excluding Serializable (a false-positive).
  private val ioExcludingSerializable: DescribedPredicate[JavaClass] =
    new DescribedPredicate[JavaClass]("reside in java.io (excluding Serializable) or java.nio.file"):
      override def test(c: JavaClass): Boolean =
        val name = c.getName
        val pkg  = c.getPackageName
        (pkg.startsWith("java.io") && name != "java.io.Serializable") ||
          pkg.startsWith("java.nio.file")

  private val inSlf4j: DescribedPredicate[JavaClass] =
    new DescribedPredicate[JavaClass]("reside in org.slf4j"):
      override def test(c: JavaClass): Boolean = c.getPackageName.startsWith("org.slf4j")

  // P0.B resolved: menger.objects no longer imports menger.input
  "menger.objects" should "depend only on menger.common (P0.B resolved)" in:
    noClasses().that().resideInAPackage("menger.objects..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("menger.input..", "menger.engines..",
                            "menger.dsl..", "menger.cli..", "menger.config..")
      .check(allClasses)

  // P0.A resolved: SceneConverter moved to menger.engines; Camera/Plane/Scene config methods inlined there.
  // menger.dsl.Material imports menger.common.Material (allowed — common is not restricted by this rule).
  it should "depend only on menger.common and menger.objects (P0.A resolved)" in:
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
        .resideInAnyPackage("menger.engines..", "io.github.lene.optix..")
      .check(allClasses)

  // Fixed: replaced haveOnlyFinalFields() with a Scala-aware var-setter check.
  // Scala val fields compile to non-final JVM fields; only var fields generate setter methods.
  "menger.common" should "have only final fields (no var)" in:
    noClasses().that().resideInAPackage("menger.common..")
      .should(haveNoVarFields)
      .check(allClasses)

  "menger.dsl" should "not use mutable collections" in:
    noClasses().that().resideInAPackage("menger.dsl..")
      .should().dependOnClassesThat()
        .resideInAPackage("scala.collection.mutable..")
      .check(allClasses)

  // Fixed: Scala case classes implement java.io.Serializable implicitly; excluded from the rule.
  "menger.common" should "not use file IO" in:
    noClasses().that().resideInAPackage("menger.common..")
      .should().dependOnClassesThat(ioExcludingSerializable)
      .check(allClasses)

  // Fixed: removed LazyLogging from Rotation (debug) and ParametricTessellator (moved to dsl layer).
  // Fixed: Scala case classes implement java.io.Serializable implicitly; excluded from the rule.
  "menger.objects" should "not use file IO or logging" in:
    noClasses().that().resideInAPackage("menger.objects..")
      .should().dependOnClassesThat(ioExcludingSerializable.or(inSlf4j))
      .check(allClasses)

  // Rule 2.4 (sealed hierarchies) is not expressible in ArchUnit — enforced by code review.
  // dsl scene-node traits and MengerException subtypes must be sealed.

  // Sprint 30.8c: engine classes must stay orchestrators, not monoliths.
  // ArchUnit works on bytecode, so we count methods as a proxy for class size.
  // InteractiveEngine was 662 lines with 30+ methods before refactoring.
  private val haveReasonableMethodCount: ArchCondition[JavaClass] =
    new ArchCondition[JavaClass]("have <= 25 methods (engines must stay lean)"):
      override def check(clazz: JavaClass, events: ConditionEvents): Unit =
        val ownMethods = clazz.getMethods.stream()
          .filter(m => m.getOwner.getName == clazz.getName)
          .count()
        val count = ownMethods.toInt
        if count > 25 then
          events.add(SimpleConditionEvent.violated(
            clazz,
            s"${clazz.getSimpleName} has $count methods (max 25). " +
            "Engines should be orchestrators — extract logic into strategy objects."
          ))

  "menger.engines" should "contain lean engine classes (<= 25 methods each)" in:
    noClasses().that().resideInAPackage("menger.engines..")
      .and().haveSimpleNameEndingWith("Engine")
      .should(haveReasonableMethodCount)
      .check(allClasses)
