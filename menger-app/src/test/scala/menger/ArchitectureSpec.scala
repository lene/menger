package menger

import com.tngtech.archunit.core.importer.ClassFileImporter
import com.tngtech.archunit.core.importer.ImportOption.DoNotIncludeTests
import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses
import com.tngtech.archunit.library.dependencies.SlicesRuleDefinition
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Architecture rules enforcing module boundaries across menger.common / menger.optix / menger-app. */
class ArchitectureSpec extends AnyFlatSpec with Matchers:

  // Import all non-test production classes on the classpath under the menger package.
  // In sbt test, all three modules are on the classpath: menger-common, optix-jni, menger-app.
  private lazy val allClasses =
    ClassFileImporter()
      .withImportOption(DoNotIncludeTests())
      .importPackages("menger")

  "menger.common" should "not depend on menger.optix" in:
    noClasses().that().resideInAPackage("menger.common..")
      .should().dependOnClassesThat().resideInAPackage("menger.optix..")
      .check(allClasses)

  it should "not depend on menger application layer" in:
    noClasses().that().resideInAPackage("menger.common..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("menger.engines..", "menger.config..", "menger.dsl..")
      .check(allClasses)

  "menger.optix" should "not depend on menger application layer" in:
    noClasses().that().resideInAPackage("menger.optix..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("menger.engines..", "menger.config..", "menger.dsl..")
      .check(allClasses)

  // Blocked by cli→engines→config→cli cycle: EnvironmentConfig holds LightSpec/PlaneConfig
  // from menger.cli. Un-ignore after Task 8 (P0.A) moves those types to menger.common.
  ignore should "be free of dependency cycles" in:
    SlicesRuleDefinition.slices()
      .matching("menger.(*)..")
      .should().beFreeOfCycles()
      .check(allClasses)
