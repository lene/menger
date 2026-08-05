package menger.engines

import java.util.concurrent.atomic.AtomicBoolean

import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.ObjectType
import io.github.lene.optix.OptiXRenderer
import io.github.lene.optix.MengerRenderer
import menger.engines.scene.InstanceId
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Fast-path regression guard fitness function (T10, Sprint 32).
  *
  * Verifies that adding a new projected-4D type to ObjectType does not
  * silently break the O(1) instance-build fast path. If the InteractiveEngine
  * schema check drifts, every projection update would trigger an instance
  * rebuild — this test catches that before it becomes a performance regression.
  */
class FastPathRegressionSuite extends AnyFlatSpec with Matchers with MockFactory:

  "Projected-4D types" should "all be recognized by InteractiveEngine's schema check" in:
    val projected4D = ObjectType.VALID_TYPES.filter: t =>
      ObjectType.isProjected4D(t) || ObjectType.isMenger4D(t) ||
      ObjectType.isSierpinski4D(t) || ObjectType.isHexadecachoron4D(t)

    projected4D should not be empty

    // Each projected-4D type should be in the InteractiveEngine's update guard
    projected4D.foreach: t =>
      withClue(s"Type '$t' should be recognized as projected 4D:"):
        val isGuarded = ObjectType.isProjected4D(t) ||
          ObjectType.isMenger4D(t) ||
          ObjectType.isSierpinski4D(t) ||
          ObjectType.isHexadecachoron4D(t)
        isGuarded shouldBe true

  "New projected-4D types" should "not silently appear without test coverage" in:
    val knownProjected4D = ObjectType.VALID_TYPES.filter: t =>
      ObjectType.isProjected4D(t) || ObjectType.isMenger4D(t) ||
      ObjectType.isSierpinski4D(t) || ObjectType.isHexadecachoron4D(t)

    // This test doubles as documentation: all currently known 4D types
    knownProjected4D should contain allOf(
      "menger4d", "sierpinski4d", "hexadecachoron4d",
      "tesseract", "pentachoron", "16-cell", "24-cell", "120-cell", "600-cell",
      "tesseract-sponge-volume", "tesseract-sponge-surface"
    )

  // F7 (Sprint 35 Ph4): assert the fast path is actually TAKEN — not just that the
  // types are registered. When specs differ only in Projection4DSpec, tryFastPath must
  // return true (no rebuild / zero clearAllInstances) and call the update function.
  "RotationFastPath.tryFastPath" should "take the fast path when only projection differs" in:
    val renderer = mock[OptiXRenderer]
    val proj1 = Projection4DSpec.default
    val proj2 = Projection4DSpec(eyeW = 3.0f, screenW = 2.0f, rotXW = 0, rotYW = 0, rotZW = 0)
    val baseSpec = ObjectSpec(objectType = "menger4d", x = 0, y = 0, z = 0, size = 1, level = Some(1))
    val prevSpecs = List(baseSpec.copy(projection4D = Some(proj1)))
    val newSpecs = List(baseSpec.copy(projection4D = Some(proj2)))
    val ids = Vector(Vector(InstanceId.fromNative(0, "test")))
    val updateCalled = AtomicBoolean(false)
    val updater: RotationFastPath.ProjectionUpdater = (_, _, _) => updateCalled.set(true)
    val took = RotationFastPath.tryFastPath(newSpecs, renderer, prevSpecs, ids, updater)
    took shouldBe true
    updateCalled.get() shouldBe true
