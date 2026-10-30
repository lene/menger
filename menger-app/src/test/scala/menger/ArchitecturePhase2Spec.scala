package menger

import com.tngtech.archunit.base.DescribedPredicate
import com.tngtech.archunit.core.domain.JavaClass
import com.tngtech.archunit.core.importer.ClassFileImporter
import com.tngtech.archunit.core.importer.ImportOption
import com.tngtech.archunit.core.importer.ImportOption.DoNotIncludeTests
import com.tngtech.archunit.core.importer.Location
import com.tngtech.archunit.lang.ArchCondition
import com.tngtech.archunit.lang.ArchRule
import com.tngtech.archunit.lang.ConditionEvents
import com.tngtech.archunit.lang.SimpleConditionEvent
import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.classes
import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object ArchitecturePhase2Spec:

  /** Layer -> the other layers it may depend on (measured 2026-10-30). `menger.common` is the
    * published base; `menger.spec` and `menger.video` are value-type leaves; `menger.optix` is
    * the renderer wrapper. A layer's own subpackages are always allowed. */
  val AllowedLayerDependencies: List[(String, List[String])] = List(
    "menger.objects" -> List("menger.common"),
    "menger.dsl"     -> List("menger.common", "menger.objects", "menger.spec", "menger.video"),
    "menger.config"  -> List("menger.common", "menger.dsl", "menger.objects", "menger.spec",
                             "menger.video"),
    "menger.input"   -> List("menger.common", "menger.objects", "menger.optix"),
    "menger.cli"     -> List("menger.common", "menger.config", "menger.dsl", "menger.spec"),
    "menger.spec"    -> List("menger.common", "menger.video"),
    "menger.scene"   -> List("menger.common", "menger.dsl"),
    "menger.video"   -> Nil
  )

  // The packages the architecture governs: ours. Everything else (java, scala, LibGDX, slf4j,
  // ...) is the business of the rules about those libraries (file IO, logging, mutable collections).
  private val governed = Seq("menger..", "io.github.lene.optix..")

  /** `layer` depends, among the governed packages, only on itself and `allowed`. Fails if the
    * layer has no classes at all, so a renamed package cannot leave the rule vacuous. */
  def layerRule(layer: String, allowed: List[String]): ArchRule =
    val permitted = (layer :: allowed).map(p => s"$p..")
    classes().that().resideInAPackage(s"$layer..")
      .should().onlyDependOnClassesThat(
        JavaClass.Predicates.resideOutsideOfPackages(governed*)
          .or(JavaClass.Predicates.resideInAnyPackage(permitted*))
      )

/**
 * Phase 2 architectural rules.
 */
class ArchitecturePhase2Spec extends AnyFlatSpec with Matchers:
  import ArchitecturePhase2Spec.*

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

  // Allow-lists, not deny-lists (menger#90 (e)): each layer names what it may depend on, so a new
  // edge fails until someone decides it belongs, instead of passing because nobody thought to
  // forbid it. The lists are the dependency graph measured 2026-10-30; ArchitectureRulesCanFailSpec
  // plants a violation in each layer and shows its rule going red.
  for (layer, allowed) <- AllowedLayerDependencies do
    s"$layer" should s"depend on no other layer than ${allowed.mkString(", ")}" in:
      layerRule(layer, allowed).check(allClasses)

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
