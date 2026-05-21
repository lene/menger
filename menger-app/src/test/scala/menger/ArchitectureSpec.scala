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

  "optix-jni public API" should "not expose Scala-specific types in method signatures" in:
    import com.tngtech.archunit.lang.{ArchCondition, ConditionEvents, SimpleConditionEvent}
    import com.tngtech.archunit.core.domain.JavaMethod
    import scala.jdk.CollectionConverters.*

    val scalaSpecificPrefixes = Set(
      "scala.Option", "scala.collection", "scala.util.Try",
      "scala.util.Either", "scala.Function"
    )
    def isScalaSpecific(typeName: String): Boolean =
      scalaSpecificPrefixes.exists(typeName.startsWith)

    // Scala compiler generates Product methods (productElement, productIterator,
    // productElementNames, canEqual) on every case class; these are not hand-written
    // API and should not trigger the rule.
    val scalaGeneratedMethodNames = Set(
      "productElement", "productIterator", "productElementNames",
      "productArity", "canEqual", "copy", "apply", "unapply"
    )

    val noScalaTypesInSignature: ArchCondition[JavaMethod] =
      new ArchCondition[JavaMethod]("not expose Scala-specific types in signatures"):
        override def check(method: JavaMethod, events: ConditionEvents): Unit =
          if scalaGeneratedMethodNames.contains(method.getName) then return
          val allTypes = method.getRawParameterTypes.asScala.toList :+ method.getRawReturnType
          allTypes.foreach: t =>
            if isScalaSpecific(t.getFullName) then
              events.add(SimpleConditionEvent.violated(method,
                s"Public method '${method.getOwner.getSimpleName}.${method.getName}' " +
                s"uses Scala-specific type '${t.getFullName}'"))

    import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.methods
    // Check only OptiXRenderer — the hand-written public JNI entry point.
    // Private[optix] traits (OptiXSphereApi, OptiXMeshApi, etc.) are implementation
    // details not accessible outside the package; they are excluded by name.
    methods().that()
      .areDeclaredInClassesThat().haveSimpleName("OptiXRenderer")
      .and().arePublic()
      .should(noScalaTypesInSignature)
      .check(allClasses)

  // Blocked by cli→engines→config→cli cycle: EnvironmentConfig holds LightSpec/PlaneConfig
  // from menger.cli. Un-ignore after Task 8 (P0.A) moves those types to menger.common.
  ignore should "be free of dependency cycles" in:
    SlicesRuleDefinition.slices()
      .matching("menger.(*)..")
      .should().beFreeOfCycles()
      .check(allClasses)
