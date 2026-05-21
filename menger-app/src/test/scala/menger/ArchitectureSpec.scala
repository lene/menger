package menger

import com.tngtech.archunit.core.importer.ClassFileImporter
import com.tngtech.archunit.core.importer.ImportOption
import com.tngtech.archunit.core.importer.ImportOption.DoNotIncludeTests
import com.tngtech.archunit.core.importer.Location
import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.{classes, noClasses}
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
    import scala.jdk.CollectionConverters._

    val scalaSpecificPrefixes = Set(
      "scala.Option", "scala.collection", "scala.util.Try",
      "scala.util.Either", "scala.Function"
    )
    def isScalaSpecific(typeName: String): Boolean =
      scalaSpecificPrefixes.exists(typeName.startsWith)

    // Scala compiler generates Product methods (productElement, productIterator,
    // productElementNames, canEqual) on every case class; PartialFunction generates
    // applyOrElse; these are not hand-written API and should not trigger the rule.
    // Methods whose name contains '$' are compiler-internal bridge/synthetic methods.
    val scalaGeneratedMethodNames = Set(
      "productElement", "productIterator", "productElementNames",
      "productArity", "canEqual", "copy", "apply", "unapply",
      "applyOrElse", "isDefinedAt", "orElse"
    )
    def isCompilerGenerated(name: String): Boolean =
      name.contains("$")

    val noScalaTypesInSignature: ArchCondition[JavaMethod] =
      new ArchCondition[JavaMethod]("not expose Scala-specific types in signatures"):
        override def check(method: JavaMethod, events: ConditionEvents): Unit =
          if scalaGeneratedMethodNames.contains(method.getName) then return
          if isCompilerGenerated(method.getName) then return
          val allTypes = method.getRawParameterTypes.asScala.toList :+ method.getRawReturnType
          allTypes.foreach: t =>
            if isScalaSpecific(t.getFullName) then
              events.add(SimpleConditionEvent.violated(method,
                s"Public method '${method.getOwner.getSimpleName}.${method.getName}' " +
                s"uses Scala-specific type '${t.getFullName}'"))

    import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.methods
    // Check all public methods in the menger.optix package.
    // Exclude Scala compiler-generated synthetic classes (companions end with "$",
    // anonymous classes contain "$anon$") which are not hand-written public API.
    methods().that()
      .areDeclaredInClassesThat().resideInAPackage("menger.optix..")
      .and().areDeclaredInClassesThat().haveSimpleNameNotEndingWith("$")
      .and().arePublic()
      .should(noScalaTypesInSignature)
      .check(allClasses)

  "production code" should "not write to standard streams (use SLF4J instead)" in:
    noClasses().that().haveNameNotMatching(".*Main.*")
      .should().accessClassesThat()
        .haveFullyQualifiedName("java.io.PrintStream")
      .check(allClasses)

  it should "not call System.exit outside Main" in:
    import com.tngtech.archunit.core.domain.JavaCall
    import com.tngtech.archunit.core.domain.AccessTarget
    import com.tngtech.archunit.base.DescribedPredicate
    // Only flag exit() declared on java.lang.System or scala.sys.package$,
    // not GDX's Application.exit() or internal GdxRuntime.exit().
    val sysExitTarget =
      new DescribedPredicate[AccessTarget.CodeUnitCallTarget]("exit on System or scala.sys"):
        override def test(t: AccessTarget.CodeUnitCallTarget): Boolean =
          t.getName == "exit" &&
            (t.getOwner.getFullName == "java.lang.System" ||
             t.getOwner.getFullName == "scala.sys.package$")
    noClasses().that().haveNameNotMatching(".*Main.*")
      .should().callMethodWhere(JavaCall.Predicates.target(sysExitTarget))
      .check(allClasses)

  it should "not contain unimplemented placeholders (???)" in:
    noClasses().should().callMethodWhere(
      com.tngtech.archunit.core.domain.JavaCall.Predicates.target(
        com.tngtech.archunit.core.domain.properties.HasName.Predicates
          .name("$qmark$qmark$qmark")))
      .check(allClasses)

  "naming conventions" should "place *Engine classes in menger.engines" in:
    classes().that().haveSimpleNameEndingWith("Engine")
      .should().resideInAPackage("menger.engines..")
      .check(allClasses)

  // Blocked by structural conflicts:
  //   menger.cli.PlaneConfig (cli→config cycle, blocked by same issue as dependency-cycle test)
  //   menger.engines.InteractiveEngine$LevelConfig (inner class, cannot move without major refactor)
  //   menger.engines.TAnimationConfig, menger.input.OrbitConfig, menger.optix.CausticsConfig/RenderConfig
  //   menger.ProfilingConfig — not yet migrated to menger.config.
  // Un-ignore after Task 8 (P0.A) resolves the cli cycle and migrates misplaced configs.
  ignore should "place *Config classes in menger.config or menger.common" in:
    classes().that().haveSimpleNameEndingWith("Config")
      .should().resideInAnyPackage("menger.config..", "menger.common..")
      .check(allClasses)

  // Blocked by structural conflicts: moving OptiX* classes to menger.optix would violate
  // existing enforced rules — OptiXRenderResources/State use LibGDX (banned from menger.optix),
  // OptiXCameraHandler/Multiplexer/KeyHandler use LibGDX Input (same constraint),
  // OptiXEngineConfig aggregates menger.config types (would create optix→config dependency,
  // banned by "optix should not depend on app layer").
  // Un-ignore after restructuring OptiX integration layer (post-Task 8).
  ignore should "keep OptiX-prefixed classes in menger.optix packages" in:
    classes().that().haveSimpleNameStartingWith("OptiX")
      .should().resideInAPackage("menger.optix..")
      .check(allClasses)

  // Blocked by cli→engines→config→cli cycle: EnvironmentConfig holds LightSpec/PlaneConfig
  // from menger.cli. Un-ignore after Task 8 (P0.A) moves those types to menger.common.
  ignore should "be free of dependency cycles" in:
    SlicesRuleDefinition.slices()
      .matching("menger.(*)..")
      .should().beFreeOfCycles()
      .check(allClasses)
