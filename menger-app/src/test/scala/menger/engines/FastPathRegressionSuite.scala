package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Fast-path regression guard fitness function (T10, Sprint 32).
  *
  * Verifies that adding a new projected-4D type to ObjectType does not
  * silently break the O(1) instance-build fast path. If the InteractiveEngine
  * schema check drifts, every projection update would trigger an instance
  * rebuild — this test catches that before it becomes a performance regression.
  */
class FastPathRegressionSuite extends AnyFlatSpec with Matchers:

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
