package menger

import com.tngtech.archunit.core.importer.ClassFileImporter
import com.tngtech.archunit.core.importer.ImportOption
import com.tngtech.archunit.core.importer.ImportOption.DoNotIncludeTests
import com.tngtech.archunit.core.importer.Location
import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses
import com.tngtech.archunit.library.dependencies.SlicesRuleDefinition
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Architecture rules enforcing module boundaries across menger.common / menger.optix / menger-app. */
class ArchitectureSpec extends AnyFlatSpec with Matchers:

  // sbt places test-classes at target/scala-X.Y.Z/test-classes/ — the extra scala version segment
  // is not matched by ArchUnit's built-in DoNotIncludeTests regex (".*/target/test-classes/.*").
  // This custom option catches both layouts.
  private val doNotIncludeSbtTestClasses: ImportOption =
    (location: Location) => !location.matches(java.util.regex.Pattern.compile(".*test-classes.*"))

  // Import all non-test production classes on the classpath under the menger package.
  // In sbt test, all three modules are on the classpath: menger-common, optix-jni, menger-app.
  private lazy val allClasses =
    ClassFileImporter()
      .withImportOption(DoNotIncludeTests())
      .withImportOption(doNotIncludeSbtTestClasses)
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

  "menger.optix JNI boundary" should "not have native methods outside optix-jni module" in:
    import com.tngtech.archunit.lang.{ArchCondition, ConditionEvents, SimpleConditionEvent}
    import com.tngtech.archunit.core.domain.{JavaClass, JavaModifier}
    val hasNativeMethod: ArchCondition[JavaClass] =
      new ArchCondition[JavaClass]("not contain native methods"):
        override def check(clazz: JavaClass, events: ConditionEvents): Unit =
          clazz.getMethods.forEach: m =>
            if m.getModifiers.contains(JavaModifier.NATIVE) then
              events.add(SimpleConditionEvent.violated(clazz,
                s"${clazz.getName}.${m.getName} is a native method outside menger.optix"))
    noClasses().that()
      .resideOutsideOfPackage("menger.optix..")
      .should(hasNativeMethod)
      .check(allClasses)

  it should "not call System.loadLibrary outside optix-jni module" in:
    import com.tngtech.archunit.core.domain.JavaCall
    import com.tngtech.archunit.core.domain.properties.HasName
    noClasses().that()
      .resideOutsideOfPackage("menger.optix..")
      .should().callMethodWhere(
        JavaCall.Predicates.target(HasName.Predicates.name("loadLibrary")))
      .check(allClasses)

  it should "not couple to LibGDX" in:
    noClasses().that()
      .resideInAPackage("menger.optix..")
      .should().dependOnClassesThat()
        .resideInAPackage("com.badlogic.gdx..")
      .check(allClasses)

  // Blocked by cli→engines→config→cli cycle: EnvironmentConfig holds LightSpec/PlaneConfig
  // from menger.cli. Un-ignore after Task 8 (P0.A) moves those types to menger.common.
  ignore should "be free of dependency cycles" in:
    SlicesRuleDefinition.slices()
      .matching("menger.(*)..")
      .should().beFreeOfCycles()
      .check(allClasses)
