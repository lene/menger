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

  private val onlyOptixJniTarget: ImportOption =
    (location: Location) => location.matches(java.util.regex.Pattern.compile(".*/optix-jni/.*"))

  private lazy val optixJniClasses =
    ClassFileImporter()
      .withImportOption(DoNotIncludeTests())
      .withImportOption(doNotIncludeSbtTestClasses)
      .withImportOption(onlyOptixJniTarget)
      .importPackages("menger.optix")

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

  // Blocked by remaining misplaced *Config types (see CODE_IMPROVEMENTS.md arch-config-naming):
  //   menger.engines.InteractiveEngine$LevelConfig — inner class; moving requires extracting to top-level
  //   menger.engines.TAnimationConfig, menger.input.OrbitConfig — wrong layer
  //   menger.optix.CausticsConfig, menger.optix.RenderConfig — optix-specific; should be menger.config
  //   menger.ProfilingConfig — root menger package, should be menger.common
  ignore should "place *Config classes in menger.config or menger.common" in:
    classes().that().haveSimpleNameEndingWith("Config")
      .should().resideInAnyPackage("menger.config..", "menger.common..")
      .check(allClasses)

  it should "keep OptiX-prefixed classes in menger.optix within optix-jni" in:
    classes().that().haveSimpleNameStartingWith("OptiX")
      .should().resideInAPackage("menger.optix..")
      .check(optixJniClasses)

  "JNI resource wrappers" should "implement AutoCloseable" in:
    import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.classes
    classes().that().haveSimpleNameContaining("Wrapper")
      .should().implement(classOf[java.lang.AutoCloseable])
      .check(allClasses)

  "domain layers" should "throw only MengerException subclasses, not raw RuntimeException" in:
    import com.tngtech.archunit.base.DescribedPredicate
    import com.tngtech.archunit.core.domain.JavaConstructorCall
    // Compiler-generated method names that legitimately call JDK exceptions:
    //   productElement / productElementName → IndexOutOfBoundsException (Scala case class synthesis)
    //   fromOrdinal / valueOf → NoSuchElementException / IllegalArgumentException (Scala enum synthesis)
    //   <init> → RuntimeException (MengerException base class calling super constructor)
    // Exclude these to avoid flagging Scala-synthesized bytecode.
    val compilerGeneratedCallerMethods = Set(
      "productElement", "productElementName", "fromOrdinal", "valueOf", "<init>"
    )
    // Flag constructor calls whose target class is a RuntimeException subtype but not a MengerException.
    val constructsRawRuntimeException: DescribedPredicate[JavaConstructorCall] =
      new DescribedPredicate[JavaConstructorCall](
        "construct RuntimeException but not a MengerException subtype"):
        override def test(call: JavaConstructorCall): Boolean =
          val callerMethod = call.getOrigin.getName
          if compilerGeneratedCallerMethods.contains(callerMethod) then return false
          val owner = call.getTarget.getOwner
          // scala.MatchError is compiler-generated for non-exhaustive patterns
          if owner.getFullName == "scala.MatchError" then return false
          owner.isAssignableTo(classOf[RuntimeException]) &&
          !owner.isAssignableTo(classOf[menger.common.MengerException])
    noClasses().that()
      .resideInAnyPackage("menger.common..", "menger.dsl..", "menger.objects..")
      .should().callConstructorWhere(constructsRawRuntimeException)
      .check(allClasses)

  it should "be free of dependency cycles" in:
    SlicesRuleDefinition.slices()
      .matching("menger.(*)..")
      .should().beFreeOfCycles()
      .check(allClasses)
