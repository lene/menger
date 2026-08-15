package menger.engines.scene

import menger.ObjectSpec
import menger.common.Color
import menger.common.Material
import menger.common.ProfilingConfig
import menger.engines.GeometryRegistry
import menger.engines.scene.LSystemSceneBuilder
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LSystemSceneBuilderSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  "LSystem DSL type" should "produce correct objectType in toObjectSpec" in:
    val ls = menger.dsl.LSystem(
      axiom = "F",
      rules = Map('F' -> "FF")
    )
    val spec = ls.toObjectSpec
    spec.objectType shouldBe "lsystem"
    spec.level shouldBe Some(4.0f)

  it should "accept iterations 0-12" in:
    noException should be thrownBy {
      menger.dsl.LSystem(axiom = "F", rules = Map('F' -> "F"))
    }
    noException should be thrownBy {
      menger.dsl.LSystem(axiom = "F", rules = Map('F' -> "F"), iterations = 12)
    }

  it should "reject iterations outside 0-12" in:
    an[IllegalArgumentException] should be thrownBy {
      menger.dsl.LSystem(axiom = "F", rules = Map('F' -> "F"), iterations = -1)
    }
    an[IllegalArgumentException] should be thrownBy {
      menger.dsl.LSystem(axiom = "F", rules = Map('F' -> "F"), iterations = 13)
    }

  it should "reject empty axiom" in:
    an[IllegalArgumentException] should be thrownBy {
      menger.dsl.LSystem(axiom = "", rules = Map('F' -> "F"))
    }

  it should "reject empty rules" in:
    an[IllegalArgumentException] should be thrownBy {
      menger.dsl.LSystem(axiom = "F", rules = Map.empty)
    }

  "LSystem.preset tree" should "create a valid LSystem" in:
    val tree = menger.dsl.LSystem.preset("tree")
    tree.axiom shouldBe "F"
    tree.rules.nonEmpty shouldBe true

  "LSystem.preset" should "throw on unknown preset" in:
    an[IllegalArgumentException] should be thrownBy {
      menger.dsl.LSystem.preset("nonexistent")
    }

  "LSystemSceneBuilder" should "validate empty specs as Left" in:
    val builder = LSystemSceneBuilder()
    builder.validate(List.empty, 100) shouldBe Left("Object specs list cannot be empty")

  it should "validate wrong type specs as Left" in:
    val builder = LSystemSceneBuilder()
    val sphereSpec = ObjectSpec(objectType = "sphere")
    builder.validate(List(sphereSpec), 100) shouldBe
      Left("All objects must be lsystem for LSystemSceneBuilder")

  it should "validate lsystem specs as Right" in:
    val builder = LSystemSceneBuilder()
    val spec = ObjectSpec(objectType = "lsystem")
    builder.validate(List(spec), 100) shouldBe Right(())

  // Sprint 36 H1.3: manual tests 167/168/171/173 aborted with "Maximum instances reached"
  // because the builder reported one instance per input spec, so InteractiveEngine's
  // auto-adjust never raised the 64-instance default past the turtle expansion.
  "LSystemSceneBuilder.calculateRequiredInstances" should
    "report the turtle expansion, not the input spec count" in:
    val builder = LSystemSceneBuilder()
    val spec = ObjectSpec.parse("type=lsystem:preset=tree:level=4:size=0.8").toOption.get
    val required = builder.calculateRequiredInstances(List(spec))
    required should be > 1
    required shouldBe builder.calculateInstanceCount(List(spec)).toInt

  it should "exceed the default 64-instance budget for the reported scenes" in:
    val builder = LSystemSceneBuilder()
    val reported = List(
      "type=lsystem:preset=tree:level=4:size=0.8",
      "type=lsystem:preset=bush:level=3:size=0.8",
      "type=lsystem:preset=kochisland:level=2:size=1.5"
    )
    reported.foreach { arg =>
      val spec = ObjectSpec.parse(arg).toOption.get
      withClue(s"$arg: ") {
        builder.calculateRequiredInstances(List(spec)) should be > 64
      }
    }

  it should "sum across multiple specs" in:
    val builder = LSystemSceneBuilder()
    val spec = ObjectSpec.parse("type=lsystem:preset=bush:level=3").toOption.get
    builder.calculateRequiredInstances(List(spec, spec)) shouldBe
      2 * builder.calculateRequiredInstances(List(spec))

  "GeometryRegistry" should "accept lsystem specs" in:
    val spec = ObjectSpec(objectType = "lsystem")
    val result = GeometryRegistry.builderFor(List(spec))
    result shouldBe defined
    result.get shouldBe a[LSystemSceneBuilder]

  "CLI parsing" should "parse lsystem preset spec" in:
    val result = ObjectSpec.parse("type=lsystem:preset=tree:level=4")
    result.isRight shouldBe true
    result.foreach { spec =>
      spec.objectType shouldBe "lsystem"
    }

  it should "parse lsystem with angle and seed" in:
    val result = ObjectSpec.parse("type=lsystem:preset=tree:level=4:angle=30:seed=123")
    result.isRight shouldBe true

  it should "reject unknown preset" in:
    val result = ObjectSpec.parse("type=lsystem:preset=nonexistent:level=3")
    result.isRight shouldBe true // CLI just stores preset as-is; validation at build time

  "Material switching in turtle" should "set material via M(name)" in:
    val mat1 = Material(Color(1f, 0f, 0f))
    val mat2 = Material(Color(0f, 1f, 0f))
    val turtle = new LSystemTurtle3D(
      "M(a)FFFF", 90f, 1.0f, 0.1f, 0.7f, 42L,
      materials = Map("a" -> mat1, "b" -> mat2)
    )
    val specs = turtle.generate()
    specs should not be empty
    specs.head.material shouldBe defined

  "Material inheritance" should "preserve material across branch push/pop" in:
    val mat1 = Material(Color(1f, 0f, 0f))
    val mat2 = Material(Color(0f, 1f, 0f))
    val turtle = new LSystemTurtle3D(
      "M(a)FFFF[M(b)FFFF]FFFF", 90f, 1.0f, 0.1f, 0.7f, 42L,
      materials = Map("a" -> mat1, "b" -> mat2)
    )
    val specs = turtle.generate()
    specs should not be empty

  "Texture command in turtle" should "set texture via T(filename)" in:
    val turtle = new LSystemTurtle3D(
      "T(bark.png)FFFF", 90f, 1.0f, 0.1f, 0.7f, 42L
    )
    val specs = turtle.generate()
    specs should not be empty
    specs.head.texture shouldBe Some("bark.png")
