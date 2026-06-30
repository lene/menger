package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.lang.reflect.Modifier

/** Module-scoped native-binding fitness function (T5, Sprint 32).
  *
  * Verifies that menger-geometry classes extend the published optix-jni
  * native surface rather than duplicating it. MengerRenderer must only
  * override native methods from OptiXRenderer, never introduce new ones.
  */
class NativeBindingModuleSuite extends AnyFlatSpec with Matchers:

  private def isNative(m: java.lang.reflect.Method): Boolean =
    Modifier.isNative(m.getModifiers)

  "MengerRenderer" should "only override native methods, not declare new ones" in:
    val cls = Class.forName("io.github.lene.optix.MengerRenderer")
    val superClass = cls.getSuperclass
    val ownNative = cls.getDeclaredMethods.filter(isNative).map(_.getName).toSet
    val superNative = superClass.getMethods.filter(isNative).map(_.getName).toSet
    val newNatives = ownNative -- superNative

    withClue(s"New @native methods in MengerRenderer: ${newNatives.mkString(", ")}"):
      newNatives shouldBe empty

  it should "extend OptiXRenderer (the published native surface)" in:
    val cls = Class.forName("io.github.lene.optix.MengerRenderer")
    cls.getSuperclass.getName should include("OptiXRenderer")

  "menger.engines" should "not declare @native methods" in:
    // Quick check: no @native in menger engine classes
    val engines = List(
      "menger.engines.GeometryRegistry",
      "menger.engines.RenderModeSelector",
      "menger.engines.TypeRegistry",
      "menger.engines.InteractiveEngine"
    )
    engines.foreach: name =>
      try
        val cls = Class.forName(name)
        val natives = cls.getDeclaredMethods.filter(isNative)
        withClue(s"@native methods in $name:"):
          natives shouldBe empty
      catch { case _: ClassNotFoundException => /* skip if not on classpath */ }
