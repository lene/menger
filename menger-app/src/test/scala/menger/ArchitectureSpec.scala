package menger

import com.tngtech.archunit.core.importer.ClassFileImporter
import com.tngtech.archunit.core.importer.ImportOption
import com.tngtech.archunit.core.importer.ImportOption.DoNotIncludeTests
import com.tngtech.archunit.core.importer.Location
import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.classes
import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses
import com.tngtech.archunit.library.dependencies.SlicesRuleDefinition
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Architecture rules enforcing module boundaries across menger.common / io.github.lene.optix / menger-app. */
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
      .importPackages("menger", "io.github.lene.optix")

  private val onlyOptixJniTarget: ImportOption =
    (location: Location) => location.matches(java.util.regex.Pattern.compile(".*/optix-jni/.*"))

  private lazy val optixJniClasses =
    ClassFileImporter()
      .withImportOption(DoNotIncludeTests())
      .withImportOption(doNotIncludeSbtTestClasses)
      .withImportOption(onlyOptixJniTarget)
      .importPackages("io.github.lene.optix")

  // menger-app's and menger-geometry's own classes in the library package (everything in
  // io.github.lene.optix that does not come from the optix-jni artifact).
  private lazy val mengerClassesInOptixPackage =
    ClassFileImporter()
      .withImportOption(DoNotIncludeTests())
      .withImportOption(doNotIncludeSbtTestClasses)
      .withImportOption((location: Location) => !onlyOptixJniTarget.includes(location))
      .importPackages("io.github.lene.optix")

  "menger.common" should "not depend on io.github.lene.optix" in:
    noClasses().that().resideInAPackage("menger.common..")
      .should().dependOnClassesThat().resideInAPackage("io.github.lene.optix..")
      .check(allClasses)

  it should "not depend on menger application layer" in:
    noClasses().that().resideInAPackage("menger.common..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("menger.engines..", "menger.config..", "menger.dsl..")
      .check(allClasses)

  // The list used to omit cli, input, tools, objects and the root package (review 2026-10-14,
  // menger#90); "menger" without ".." is the root package itself.
  "io.github.lene.optix" should "not depend on menger application layer" in:
    noClasses().that().resideInAPackage("io.github.lene.optix..")
      .should().dependOnClassesThat()
        .resideInAnyPackage("menger.engines..", "menger.config..", "menger.dsl..",
          "menger.cli..", "menger.input..", "menger.tools..", "menger.objects..", "menger")
      .check(allClasses)

  "io.github.lene.optix JNI boundary" should "not have native methods outside optix-jni module" in:
    import com.tngtech.archunit.lang.{ArchCondition, ConditionEvents, SimpleConditionEvent}
    import com.tngtech.archunit.core.domain.{JavaClass, JavaModifier}
    val hasNativeMethod: ArchCondition[JavaClass] =
      new ArchCondition[JavaClass]("not contain native methods"):
        override def check(clazz: JavaClass, events: ConditionEvents): Unit =
          clazz.getMethods.forEach: m =>
            if m.getModifiers.contains(JavaModifier.NATIVE) then
              events.add(SimpleConditionEvent.violated(clazz,
                s"${clazz.getName}.${m.getName} is a native method outside io.github.lene.optix"))
    noClasses().that()
      .resideOutsideOfPackage("io.github.lene.optix..")
      .should(hasNativeMethod)
      .check(allClasses)

  it should "not call System.loadLibrary outside optix-jni module" in:
    import com.tngtech.archunit.core.domain.JavaCall
    import com.tngtech.archunit.core.domain.properties.HasName
    noClasses().that()
      .resideOutsideOfPackage("io.github.lene.optix..")
      .should().callMethodWhere(
        JavaCall.Predicates.target(HasName.Predicates.name("loadLibrary")))
      .check(allClasses)

  it should "not couple to LibGDX" in:
    noClasses().that()
      .resideInAPackage("io.github.lene.optix..")
      .should().dependOnClassesThat()
        .resideInAPackage("com.badlogic.gdx..")
      .check(allClasses)

  "application layers outside input/engines" should "not depend on LibGDX" in:
    import com.tngtech.archunit.base.DescribedPredicate
    import com.tngtech.archunit.core.domain.JavaClass
    // AD-23's containment boundary: LibGDX (windowing/input adapters, LWJGL lifecycle)
    // is confined to menger.input and menger.engines. The CameraConfig/Vec3 Vector3
    // cascade (F10, M-arch-libgdx-containment) is fully migrated — the exception list
    // is down to its one permanent member: Main.scala is the LWJGL app bootstrap.
    // (RenderState/OptiXRenderResources hold genuine GL primitives, Pixmap/Texture/SpriteBatch,
    // and moved into menger.input, so they need no exception any more.) Matched by source
    // file name — Scala 3 compiles `given`/top-level definitions to synthetic class names
    // that aren't worth predicting here.
    val deferredLibGdxFiles = Set("Main.scala")
    val isDeferredLibGdxFile: DescribedPredicate[JavaClass] =
      new DescribedPredicate[JavaClass]("declared in a file with accepted LibGDX debt"):
        override def test(clazz: JavaClass): Boolean =
          val fileName = clazz.getSource.map[String](s => s.getFileName.orElse("")).orElse("")
          deferredLibGdxFiles.contains(fileName)
    noClasses().that()
      .resideOutsideOfPackage("menger.input..")
      .and().resideOutsideOfPackage("menger.engines..")
      .and(DescribedPredicate.not(isDeferredLibGdxFile))
      .should().dependOnClassesThat().resideInAPackage("com.badlogic.gdx..")
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
    // Check all public methods in the io.github.lene.optix package.
    // Exclude Scala compiler-generated synthetic classes (companions end with "$",
    // anonymous classes contain "$anon$") which are not hand-written public API.
    methods().that()
      .areDeclaredInClassesThat().resideInAPackage("io.github.lene.optix..")
      .and().areDeclaredInClassesThat().haveSimpleNameNotEndingWith("$")
      .and().arePublic()
      .should(noScalaTypesInSignature)
      .check(optixJniClasses)

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

  it should "place *Config classes in menger.config or menger.common" in:
    classes().that().haveSimpleNameEndingWith("Config")
      .should().resideInAnyPackage("menger.config..", "menger.common..")
      .check(allClasses)

  // Split package (review 2026-10-14, menger#89/#90): menger's own classes declared the published
  // library's package io.github.lene.optix, which let them reach optix-jni's private[optix]
  // members. They moved to menger.optix and menger.geometry; a test hook comes from
  // io.github.lene.optix.testing.OptiXTestHooks (optix-jni 0.5.0). None may come back.
  it should "declare no menger classes in the library package io.github.lene.optix" in:
    import scala.jdk.CollectionConverters.*
    mengerClassesInOptixPackage.asScala.map(_.getName).toList.sorted shouldBe empty

  // The same split in the other direction (menger#89): menger.common belongs to the published
  // menger-common artifact; ObjectType, the one menger-app class in it, moved to menger.spec.
  it should "declare no menger-app classes in the library package menger.common" in:
    import scala.jdk.CollectionConverters.*
    val mengerClassesInCommonPackage = ClassFileImporter()
      .withImportOption(DoNotIncludeTests())
      .withImportOption(doNotIncludeSbtTestClasses)
      .withImportOption((location: Location) => !location.isJar)
      .importPackages("menger.common")
    mengerClassesInCommonPackage.asScala.map(_.getName).toList.sorted shouldBe empty

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

  // The root package `menger` belongs to no layer and no slice, so the cycle rule above cannot
  // see the two cycles that run through it (review 2026-10-14, menger#84/#90). Until it is
  // emptied (menger#84), it may not grow: a new type there fails until listed here deliberately.
  it should "not grow the unlayered root package menger" in:
    classes().that().resideInAPackage("menger")
      .should().haveNameMatching(
        "menger\\.GlobalRotation(\\$.*)?"
      ).check(allClasses)

  it should "keep menger.video a leaf package" in:
    noClasses().that().resideInAPackage("menger.video..")
      .should().dependOnClassesThat().resideInAnyPackage(
        "menger.engines..", "menger.dsl..", "menger.objects.."
      )
      .check(allClasses)

  // SA-AD-4 rule 2: `RestrictedClasspath` hands the compiler only the DSL surface, so a scene
  // file cannot reach the rest. That is only safe while the DSL surface itself does not depend
  // on the rest -- if it ever does, scene compilation breaks with an opaque "class not found".
  // menger.scene (the compiler and loader, moved out of menger.dsl) and the compiler itself
  // are in the list since review 2026-10-14 (menger#79): the vocabulary must not contain the
  // machinery that executes it.
  "The DSL surface" should "not depend on the engine, tooling, CLI, input or scene-loading layers" in:
    noClasses().that().resideInAnyPackage("menger.dsl..", "menger.objects..", "menger.video..")
      .should().dependOnClassesThat().resideInAnyPackage(
        "menger.engines..", "menger.tools..", "menger.cli..", "menger.input..", "menger.scene..",
        "dotty.."
      )
      .check(allClasses)

  // `MeshFactory.mesh4D` was widened from private so SceneValidator can run
  // PolytopeInvariants against the same Mesh4D the renderer builds (review round 2). The rule
  // used to forbid only menger.engines.internal -- a package that does not exist -- so it could
  // not fail while SceneValidator and SceneSnapshot reached five engine-root types (review
  // 2026-10-14, menger#90). It now admits menger.engines.scene plus exactly today's edges; a
  // new one fails until it is listed here deliberately. The validator's build checks moved into
  // BuildPrecheck (menger#101), which replaced four of the five edges with one.
  private val toolsEngineEdges = Set("BuildPrecheck", "SceneConverter").map("menger.engines." + _)

  "menger.tools" should "reach the engine layer only via engines.scene and the known edges" in:
    import com.tngtech.archunit.base.DescribedPredicate
    import com.tngtech.archunit.core.domain.JavaClass
    val unlistedEngineClass: DescribedPredicate[JavaClass] =
      new DescribedPredicate[JavaClass]("an engine class outside engines.scene and known edges"):
        override def test(c: JavaClass): Boolean =
          val name = c.getFullName
          name.startsWith("menger.engines.") && !name.startsWith("menger.engines.scene.") &&
            !toolsEngineEdges.exists(edge => name == edge || name.startsWith(edge + "$"))
    noClasses().that().resideInAPackage("menger.tools..")
      .should().dependOnClassesThat(unlistedEngineClass)
      .orShould().dependOnClassesThat().resideInAPackage("menger.input..")
      .check(allClasses)
