# ArchUnit Rules — Expanded Enforcement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand `ArchitectureSpec` from 3 module-boundary rules to a full architectural rule suite covering acyclicity, JNI isolation, naming, immutability, and layer ordering — phased to match prerequisite refactors.

**Architecture:** Phase 1 rules require no production code changes and are added directly. Phase 0 performs two decoupling refactors (dsl→cli, objects→input). Phase 2 rules are added as `@Ignore` immediately (to capture intent) then un-ignored after Phase 0 refactors complete.

**Tech Stack:** Scala 3, ScalaTest, ArchUnit 1.x (`com.tngtech.archunit`), sbt multi-module project (`menger-common`, `optix-jni`, `menger-app`).

---

## File map

| File | Action |
|------|--------|
| `menger-app/src/test/scala/menger/ArchitectureSpec.scala` | Modify — add all Phase 1 rules and Phase 2 @Ignore stubs |
| `menger-app/src/test/scala/menger/ArchitecturePhase2Spec.scala` | Create — Phase 2 rules (un-ignored after Phase 0 refactors) |
| Types to move in Phase 0.A (identified in Task 8) | Modify — move CLI-spec types to `menger.common` |
| Coupling class found in Phase 0.B (identified in Task 9) | Modify — move interactive behaviour to `menger.engines` |

---

## Reference: running tests

```bash
# Run only arch tests (fast, no compilation of CUDA/JNI):
sbt "menger-app/testOnly menger.ArchitectureSpec"
sbt "menger-app/testOnly menger.ArchitecturePhase2Spec"

# Run full test suite (use after refactors to check nothing broke):
sbt "menger-app/test"

# Run pre-push DoD gate (required before every push):
./.git_hooks/pre-push
```

## Reference: ArchUnit imports needed

```scala
import com.tngtech.archunit.core.importer.ClassFileImporter
import com.tngtech.archunit.core.importer.ImportOption.DoNotIncludeTests
import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.{noClasses, classes, methods}
import com.tngtech.archunit.library.dependencies.SlicesRuleDefinition
import com.tngtech.archunit.library.GeneralCodingRules
import com.tngtech.archunit.lang.{ArchCondition, ConditionEvents, SimpleConditionEvent}
import org.scalatest.Ignore
```

---

## Task 1: Acyclic dependencies rule (Phase 1.1)

**Files:**
- Modify: `menger-app/src/test/scala/menger/ArchitectureSpec.scala`

- [ ] **Step 1: Add import and rule to ArchitectureSpec**

  Add to the imports block at the top:
  ```scala
  import com.tngtech.archunit.library.dependencies.SlicesRuleDefinition
  ```

  Add this test block inside the class, after the existing `"menger.optix"` block:
  ```scala
  "menger packages" should "be free of dependency cycles" in:
    SlicesRuleDefinition.slices()
      .matching("menger.(*)..")
      .should().beFreeOfCycles()
      .check(allClasses)
  ```

- [ ] **Step 2: Run the rule**

  ```bash
  sbt "menger-app/testOnly menger.ArchitectureSpec"
  ```

  Expected: PASS (cycles almost certainly absent). If FAIL: the output names the cycle. Investigate with `grep -r "import menger\." src/main/scala` to trace the loop, break it by extracting the shared type, then re-run until passing.

- [ ] **Step 3: Commit**

  ```bash
  git add menger-app/src/test/scala/menger/ArchitectureSpec.scala
  git commit -m "test(arch): enforce acyclic package dependencies"
  ```

---

## Task 2: JNI boundary isolation rules (Phase 1.2)

**Files:**
- Modify: `menger-app/src/test/scala/menger/ArchitectureSpec.scala`

These rules ensure the `menger.optix` wrapper in `menger-app` is the only touch point to native code — prerequisite for `optix-jni` extraction.

- [ ] **Step 1: Add JNI boundary tests**

  Add this block to `ArchitectureSpec`:
  ```scala
  "menger.optix JNI boundary" should "be accessed only through the wrapper package" in:
    noClasses().that()
      .resideOutsideOfPackage("menger.optix..")
      .should().dependOnClassesThat()
        .resideInAPackage("menger.optix..")
        .and().areAnnotatedWith(classOf[java.lang.annotation.Native])
      .check(allClasses)

  it should "not load native libraries outside optix-jni module" in:
    noClasses().that()
      .resideOutsideOfPackage("menger.optix..")
      .should().callMethodWhere:
        com.tngtech.archunit.lang.conditions.ArchConditions
          .callMethod(classOf[java.lang.System], "loadLibrary", classOf[String])
      .check(allClasses)

  it should "not couple to LibGDX inside optix-jni" in:
    noClasses().that()
      .resideInAPackage("menger.optix..")
      .should().dependOnClassesThat()
        .resideInAPackage("com.badlogic.gdx..")
      .check(allClasses)
  ```

  > **Note on `loadLibrary` check:** ArchUnit's `callMethodWhere` predicate can be written more simply using `com.tngtech.archunit.lang.syntax.ArchRuleDefinition`'s fluent API. If the above does not compile, use this alternative:
  > ```scala
  > import com.tngtech.archunit.base.DescribedPredicate.describe
  > import com.tngtech.archunit.core.domain.JavaCall
  > noClasses().that().resideOutsideOfPackage("menger.optix..")
  >   .should(com.tngtech.archunit.lang.conditions.ArchConditions
  >     .callMethodWhere(JavaCall.Predicates.target(
  >       com.tngtech.archunit.core.domain.properties.HasName.Predicates.name("loadLibrary"))))
  >   .check(allClasses)
  > ```

- [ ] **Step 2: Run the rules**

  ```bash
  sbt "menger-app/testOnly menger.ArchitectureSpec"
  ```

  Expected: all three PASS. If any FAIL, the output names the offending class. Fix: move the class to `menger.optix` or remove the prohibited dependency.

- [ ] **Step 3: Commit**

  ```bash
  git add menger-app/src/test/scala/menger/ArchitectureSpec.scala
  git commit -m "test(arch): enforce JNI boundary isolation rules"
  ```

---

## Task 3: Java-friendly API surface rule (Phase 1.3)

**Files:**
- Modify: `menger-app/src/test/scala/menger/ArchitectureSpec.scala`

`optix-jni` is intended for any JVM language consumer (Java, Kotlin). Public method signatures must not expose Scala-specific types.

- [ ] **Step 1: Add custom ArchCondition and rule**

  Add this block to `ArchitectureSpec`. The custom condition inspects each public method's parameter and return types:

  ```scala
  private val scalaOnlyPackages = Set(
    "scala.collection", "scala.Option", "scala.util.Try",
    "scala.util.Either", "scala.Function"
  )

  private val noScalaTypesInPublicSignature: ArchCondition[JavaMethod] =
    new ArchCondition[JavaMethod]("not expose Scala-specific types in public signatures"):
      override def check(method: JavaMethod, events: ConditionEvents): Unit =
        if method.getModifiers.contains(com.tngtech.archunit.core.domain.JavaModifier.PUBLIC) then
          val allTypes = method.getRawParameterTypes.asScala.toList :+ method.getRawReturnType
          allTypes.foreach: t =>
            val name = t.getFullName
            if scalaOnlyPackages.exists(pkg => name.startsWith(pkg)) then
              events.add(SimpleConditionEvent.violated(method,
                s"Public method '${method.getName}' uses Scala-specific type '$name'"))

  "optix-jni public API" should "not expose Scala-specific types in method signatures" in:
    methods().that()
      .areDeclaredInClassesThat().resideInAPackage("menger.optix..")
      .and().arePublic()
      .should(noScalaTypesInPublicSignature)
      .check(allClasses)
  ```

  Also add to imports:
  ```scala
  import com.tngtech.archunit.core.domain.JavaMethod
  import com.tngtech.archunit.lang.{ArchCondition, ConditionEvents, SimpleConditionEvent}
  import scala.jdk.CollectionConverters.*
  ```

- [ ] **Step 2: Run the rule**

  ```bash
  sbt "menger-app/testOnly menger.ArchitectureSpec"
  ```

  Expected: PASS if `optix-jni` already uses primitive/array API. If FAIL: the output names each offending method. Fix by changing the method signature to use primitives, `Array[T]`, or `menger.common` types. Do not change internal implementation — only the public signature.

- [ ] **Step 3: Commit**

  ```bash
  git add menger-app/src/test/scala/menger/ArchitectureSpec.scala
  git commit -m "test(arch): enforce Java-friendly public API in optix-jni"
  ```

---

## Task 4: Side-effect anti-patterns (Phase 1.4)

**Files:**
- Modify: `menger-app/src/test/scala/menger/ArchitectureSpec.scala`

- [ ] **Step 1: Add rules**

  Add to `ArchitectureSpec`:
  ```scala
  "production code" should "not write to standard streams (use SLF4J instead)" in:
    noClasses().that().haveSimpleNameNotContaining("Main")
      .should().accessClassesThat()
        .haveFullyQualifiedName("java.io.PrintStream")
      .check(allClasses)

  it should "not call System.exit outside Main" in:
    noClasses().that().haveSimpleNameNotContaining("Main")
      .should().callMethodWhere:
        com.tngtech.archunit.lang.conditions.ArchConditions
          .callMethod(classOf[java.lang.System], "exit", classOf[Int])
      .check(allClasses)

  it should "not contain unimplemented placeholders (???)" in:
    noClasses().should().callMethodWhere:
      com.tngtech.archunit.core.domain.JavaCall.Predicates.target(
        com.tngtech.archunit.core.domain.properties.HasName.Predicates
          .name("$qmark$qmark$qmark"))
    .check(allClasses)
  ```

  > **Note on `???`:** In Scala 3 bytecode, `???` compiles to a call to the method named `$qmark$qmark$qmark` on `scala.Predef$`. The `name("$qmark$qmark$qmark")` predicate matches it uniquely.

- [ ] **Step 2: Run the rules**

  ```bash
  sbt "menger-app/testOnly menger.ArchitectureSpec"
  ```

  Expected: all PASS. If `println` violations: replace with `logger.debug(...)` (add `LazyLogging` mixin if not present). If `System.exit` outside Main: refactor to throw `MengerException` and let Main handle exit.

- [ ] **Step 3: Commit**

  ```bash
  git add menger-app/src/test/scala/menger/ArchitectureSpec.scala
  git commit -m "test(arch): ban println, System.exit outside Main, and ??? in production"
  ```

---

## Task 5: Naming convention rules (Phase 1.5)

**Files:**
- Modify: `menger-app/src/test/scala/menger/ArchitectureSpec.scala`

- [ ] **Step 1: Add naming rules**

  Add to `ArchitectureSpec`:
  ```scala
  "naming conventions" should "place *Engine classes in menger.engines" in:
    classes().that().haveSimpleNameEndingWith("Engine")
      .should().resideInAPackage("menger.engines..")
      .check(allClasses)

  it should "place *Config classes in menger.config or menger.common" in:
    classes().that().haveSimpleNameEndingWith("Config")
      .should().resideInAnyPackage("menger.config..", "menger.common..")
      .check(allClasses)

  it should "keep OptiX-prefixed classes in optix-jni within menger.optix" in:
    classes().that()
      .resideInAPackage("menger.optix..")
      .and().haveSimpleNameStartingWith("OptiX")
      .should().resideInAPackage("menger.optix..")
      .check(allClasses)
  ```

  Add import:
  ```scala
  import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.classes
  ```

- [ ] **Step 2: Run the rules**

  ```bash
  sbt "menger-app/testOnly menger.ArchitectureSpec"
  ```

  Expected: PASS. If `*Engine` violations: the class is misplaced — move it to `menger.engines`. If `*Config` violations: move to `menger.config` or `menger.common`. Fix each violation, then re-run.

- [ ] **Step 3: Commit**

  ```bash
  git add menger-app/src/test/scala/menger/ArchitectureSpec.scala
  git commit -m "test(arch): enforce *Engine, *Config, OptiX* naming conventions"
  ```

---

## Task 6: Resource lifecycle and error handling rules (Phase 1.6 + 1.7)

**Files:**
- Modify: `menger-app/src/test/scala/menger/ArchitectureSpec.scala`

- [ ] **Step 1: Add resource lifecycle rule**

  Add to `ArchitectureSpec`:
  ```scala
  "JNI resource wrappers" should "implement AutoCloseable" in:
    classes().that().haveSimpleNameContaining("Wrapper")
      .should().implement(classOf[java.lang.AutoCloseable])
      .check(allClasses)
  ```

- [ ] **Step 2: Add error handling rule**

  Add these imports to `ArchitectureSpec`:
  ```scala
  import com.tngtech.archunit.base.DescribedPredicate
  import com.tngtech.archunit.core.domain.JavaClass as JClass
  ```

  Add this test block to `ArchitectureSpec`:
  ```scala
  "domain layers" should "throw only MengerException subclasses, not raw RuntimeException" in:
    val isRawRuntimeException: DescribedPredicate[JClass] =
      DescribedPredicate.describe("is RuntimeException but not a MengerException subtype"):
        c => c.isAssignableTo(classOf[RuntimeException]) &&
             !c.isAssignableTo(classOf[menger.common.MengerException])
    noClasses().that()
      .resideInAnyPackage("menger.common..", "menger.dsl..", "menger.objects..")
      .should().constructClassesThat(isRawRuntimeException)
      .check(allClasses)
  ```

  > **Note:** `constructClassesThat` checks which exception classes domain code instantiates with `new`. The predicate passes for `RuntimeException` and its non-MengerException subtypes only. Direct `MengerException` subclasses are permitted.

- [ ] **Step 3: Run both rules**

  ```bash
  sbt "menger-app/testOnly menger.ArchitectureSpec"
  ```

  If `Wrapper` class doesn't implement `AutoCloseable`: add `extends AutoCloseable` and implement `def close(): Unit = dispose()` (or rename `dispose` to `close`). If domain class constructs raw `RuntimeException`: replace with the appropriate `MengerException` subclass.

- [ ] **Step 4: Commit**

  ```bash
  git add menger-app/src/test/scala/menger/ArchitectureSpec.scala
  git commit -m "test(arch): enforce AutoCloseable on wrappers and typed exceptions in domain"
  ```

---

## Task 7: Add Phase 2 rules as @Ignore stubs

**Files:**
- Create: `menger-app/src/test/scala/menger/ArchitecturePhase2Spec.scala`

These rules capture the full onion-layer intent. They are `@Ignore`d now because Phase 0 refactors (Tasks 8 + 9) must complete first. The `@Ignore` annotation makes the intent visible in the test suite without breaking CI.

- [ ] **Step 1: Create ArchitecturePhase2Spec.scala**

  ```scala
  package menger

  import com.tngtech.archunit.core.importer.ClassFileImporter
  import com.tngtech.archunit.core.importer.ImportOption.DoNotIncludeTests
  import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses
  import org.scalatest.Ignore
  import org.scalatest.flatspec.AnyFlatSpec
  import org.scalatest.matchers.should.Matchers

  /**
   * Phase 2 architectural rules — blocked on Phase 0 refactors.
   * P0.A (dsl→cli decoupling) and P0.B (objects→input decoupling) must complete first.
   * Remove @Ignore from each test once the corresponding blocker is resolved.
   */
  @Ignore
  class ArchitecturePhase2Spec extends AnyFlatSpec with Matchers:

    private lazy val allClasses =
      ClassFileImporter()
        .withImportOption(DoNotIncludeTests())
        .importPackages("menger")

    // Blocked by P0.B: menger.objects currently imports menger.input
    "menger.objects" should "depend only on menger.common (P0.B blocker)" in:
      noClasses().that().resideInAPackage("menger.objects..")
        .should().dependOnClassesThat()
          .resideInAnyPackage("menger.input..", "menger.engines..",
                              "menger.dsl..", "menger.cli..", "menger.config..")
        .check(allClasses)

    // Blocked by P0.A: menger.dsl currently imports menger.cli
    "menger.dsl" should "depend only on menger.common and menger.objects (P0.A blocker)" in:
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

    "menger.cli" should "not depend on engines, dsl, or optix" in:
      noClasses().that().resideInAPackage("menger.cli..")
        .should().dependOnClassesThat()
          .resideInAnyPackage("menger.engines..", "menger.optix..")
        .check(allClasses)

    // Immutability rules — audit required before un-ignoring
    "menger.common types" should "not contain mutable fields" in:
      noClasses().that().resideInAPackage("menger.common..")
        .should().haveOnlyFinalFields()
        .check(allClasses)

    "menger.dsl types" should "not use mutable collections" in:
      noClasses().that().resideInAPackage("menger.dsl..")
        .should().dependOnClassesThat()
          .resideInAPackage("scala.collection.mutable..")
        .check(allClasses)

    // Purity rules — audit required before un-ignoring
    "menger.common" should "not use file IO" in:
      noClasses().that().resideInAPackage("menger.common..")
        .should().dependOnClassesThat()
          .resideInAnyPackage("java.io..", "java.nio.file..")
        .check(allClasses)

    "menger.objects" should "not use file IO or logging" in:
      noClasses().that().resideInAPackage("menger.objects..")
        .should().dependOnClassesThat()
          .resideInAnyPackage("java.io..", "java.nio.file..", "org.slf4j..")
        .check(allClasses)
  ```

  > **Rule 2.4 — Sealed hierarchies:** ArchUnit operates on JVM bytecode and cannot inspect the Scala `sealed` modifier (a compile-time constraint erased at bytecode level). Rule 2.4 (scene node variants in `menger.dsl` and `MengerException` subtypes in `menger.common` must be sealed) is enforced by code review only. Add a comment to `ArchitecturePhase2Spec` to document this:
  >
  > ```scala
  > // Rule 2.4 (sealed hierarchies) is not expressible in ArchUnit — enforced by code review.
  > // dsl scene-node traits and MengerException subtypes must be sealed.
  > ```

- [ ] **Step 2: Run to confirm @Ignore works (all should be skipped)**

  ```bash
  sbt "menger-app/testOnly menger.ArchitecturePhase2Spec"
  ```

  Expected output: all tests skipped (no FAIL, no PASS — ScalaTest reports them as ignored).

- [ ] **Step 3: Commit**

  ```bash
  git add menger-app/src/test/scala/menger/ArchitecturePhase2Spec.scala
  git commit -m "test(arch): add Phase 2 layer rules as @Ignore stubs pending refactors"
  ```

---

## Task 8: Phase 0.A — Decouple menger.dsl from menger.cli

**Files:**
- Investigate: which types in `menger.cli` are imported by `menger.dsl`
- Modify: move those types to `menger.common`
- Modify: update all import sites

This is the prerequisite for un-ignoring the `menger.dsl` rule in `ArchitecturePhase2Spec`.

- [ ] **Step 1: Identify the coupling types**

  ```bash
  grep -rn "^import menger\.cli\." menger-app/src/main/scala/menger/dsl/ --include="*.scala"
  ```

  This lists every `menger.cli.*` symbol imported by DSL files. Note each type name and the file that defines it (search `menger-app/src/main/scala/menger/cli/`).

- [ ] **Step 2: Assess each type**

  For each CLI type imported by DSL, decide:
  - Is it a domain concept (e.g. `LightSpec`, `PlaneConfig`, `AxisSpec`)? → move to `menger.common`
  - Is it truly CLI-only (relates to argument parsing, not domain)? → if so, re-design DSL not to import it

  Move domain-concept types: cut them from their file in `menger/cli/`, create new files in `menger-app/src/main/scala/menger/common/` (e.g. `LightSpec.scala`), paste the type definition, change the `package` declaration from `menger.cli` to `menger.common`.

- [ ] **Step 3: Update all import sites**

  ```bash
  grep -rn "import menger\.cli\.<TypeName>" menger-app/src/main/scala/ --include="*.scala"
  ```

  Replace each `import menger.cli.<TypeName>` with `import menger.common.<TypeName>` in every file that references the moved type. Repeat for each moved type.

- [ ] **Step 4: Compile to verify**

  ```bash
  sbt "menger-app/compile"
  ```

  Expected: clean compile. Fix any remaining import errors.

- [ ] **Step 5: Run full tests**

  ```bash
  sbt "menger-app/test"
  ```

  Expected: all tests pass. Fix any test import errors the same way.

- [ ] **Step 6: Remove @Ignore from the dsl rule in ArchitecturePhase2Spec**

  In `ArchitecturePhase2Spec.scala`, remove `@Ignore` from the class annotation and re-run to verify the dsl rule now passes. If the class-level `@Ignore` also covers unrelated rules, switch to per-test `ignore` keyword instead:

  ```scala
  // Change class-level @Ignore to per-test ignore for tests still blocked:
  ignore should "..." // keep ignored
  it should "depend only on menger.common and menger.objects..." // un-ignore the dsl rule
  ```

  Run:
  ```bash
  sbt "menger-app/testOnly menger.ArchitecturePhase2Spec"
  ```

  Expected: dsl rule PASS, others still skipped.

- [ ] **Step 7: Commit**

  ```bash
  git add menger-app/src/main/scala/menger/common/ menger-app/src/main/scala/menger/cli/ \
          menger-app/src/main/scala/menger/dsl/ menger-app/src/test/scala/menger/ArchitecturePhase2Spec.scala
  git commit -m "refactor(arch): move shared CLI↔DSL types to menger.common; enable dsl layer rule"
  ```

---

## Task 9: Phase 0.B — Decouple menger.objects from menger.input

**Files:**
- Investigate: which class in `menger.objects` imports `menger.input`
- Modify: extract interactive behaviour to `menger.engines` (or `menger.engines.scene`)
- Modify: update callers

- [ ] **Step 1: Identify the coupling class**

  ```bash
  grep -rn "^import menger\.input\." menger-app/src/main/scala/menger/objects/ --include="*.scala"
  ```

  Note: the class name, what it imports from `menger.input`, and how it uses it.

- [ ] **Step 2: Separate geometry data from interactive behaviour**

  The class likely contains both pure geometry data (stays in `menger.objects`) and input-response logic (moves out). Strategy:

  a. Extract the input-dependent behaviour into a new class in `menger-app/src/main/scala/menger/engines/` (name it `<OriginalName>Controller.scala` or similar).
  b. The original class in `menger.objects` retains only pure geometry fields and methods, with no `import menger.input.*`.
  c. Callers of the extracted behaviour update their references to the new location.

- [ ] **Step 3: Compile**

  ```bash
  sbt "menger-app/compile"
  ```

  Fix any import errors.

- [ ] **Step 4: Run full tests**

  ```bash
  sbt "menger-app/test"
  ```

  Expected: all tests pass. If a test breaks, fix its import to point to the new class location.

- [ ] **Step 5: Un-ignore the objects rule in ArchitecturePhase2Spec**

  Change the `menger.objects` test from `ignore` to `it`:
  ```scala
  it should "depend only on menger.common (P0.B blocker)" in:
    ...
  ```

  Run:
  ```bash
  sbt "menger-app/testOnly menger.ArchitecturePhase2Spec"
  ```

  Expected: objects rule PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add menger-app/src/main/scala/menger/objects/ menger-app/src/main/scala/menger/engines/ \
          menger-app/src/test/scala/menger/ArchitecturePhase2Spec.scala
  git commit -m "refactor(arch): decouple menger.objects from menger.input; enable objects layer rule"
  ```

---

## Task 10: Phase 2 — Audit and enable remaining rules

After Tasks 8 and 9, several rules in `ArchitecturePhase2Spec` are still ignored (config, input, cli layer rules, immutability, purity). Enable them one at a time.

- [ ] **Step 1: Enable and run each remaining rule in turn**

  For each still-ignored rule in `ArchitecturePhase2Spec`:
  1. Change `ignore` to `it`
  2. Run: `sbt "menger-app/testOnly menger.ArchitecturePhase2Spec"`
  3. If PASS → leave enabled, move to next rule
  4. If FAIL → fix the violation (see rule-specific guidance below), then re-run

  Rule-specific fix guidance:

  **`menger.config` depends on engines or cli:**  
  Move the offending class or break the import by introducing a common interface in `menger.common`.

  **`menger.input` depends on engines/dsl/config:**  
  Extract the dependency into a callback/interface in `menger.common` that `menger.input` can depend on.

  **`menger.cli` depends on engines/optix:**  
  CLI should only parse → build config/DSL nodes. If it calls into engines, extract that call to `Main.scala`.

  **`menger.common` has mutable fields (`haveOnlyFinalFields` fails):**  
  Change `var` to `val`. If the class truly needs mutation, it belongs in `menger.engines`, not `menger.common`.

  **`menger.dsl` uses `scala.collection.mutable`:**  
  Replace with `List`, `Map`, `Set` (immutable). If building incrementally, use a local `var` builder inside a method body (not a field).

  **`menger.common` or `menger.objects` uses file IO:**  
  Move IO to a boundary method in `menger.engines` and pass the result into the domain method.

  **`menger.objects` uses SLF4J logging:**  
  Remove `LazyLogging` mixin; delete log calls. Geometry is pure data — no observable side effects.

- [ ] **Step 2: Remove @Ignore from ArchitecturePhase2Spec class if all rules now pass**

  Once all rules pass individually, remove class-level `@Ignore` annotation and run the full suite:

  ```bash
  sbt "menger-app/testOnly menger.ArchitecturePhase2Spec"
  ```

  Expected: all tests PASS.

- [ ] **Step 3: Run pre-push DoD gate**

  ```bash
  ./.git_hooks/pre-push
  ```

  Expected: all checks pass.

- [ ] **Step 4: Commit**

  ```bash
  git add menger-app/src/test/scala/menger/ArchitecturePhase2Spec.scala \
          menger-app/src/main/scala/menger/
  git commit -m "test(arch): enable all Phase 2 onion-layer and immutability rules"
  ```

---

## Phase 3 note (deferred)

Phase 3 rules (wrapper API purity, post-extraction module boundary) are not part of this plan. They are triggered when `optix-jni` is extracted as a separate Maven/sbt artifact. At that point:
- Add a rule verifying `menger.optix` wrapper public signatures return only `menger.common` types or primitives (no raw `Long` handles).
- Replace the source-classpath JNI boundary rule (Task 2) with an artifact-level dependency rule.
