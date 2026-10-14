package menger.engines

import java.io.File
import java.nio.file.Files

import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Usability review 2026-10, session 4 (F96): the rotation added in the window used to be
  * written into the specs, which a live reload overwrote. */
class Interactive4DOffsetSuite extends AnyFlatSpec with Matchers:

  private val tesseract = ObjectSpec(
    "tesseract", projection4D = Some(Projection4DSpec(rotXW = 10f, rotYW = 20f, rotZW = 0f))
  )
  private val cube = ObjectSpec("cube")

  "Interactive4DOffset.applyTo" should "add the rotation to every 4D object's projection only" in:
    val offset = Interactive4DOffset().plus(5f, -3f, 2f)

    val applied = offset.applyTo(List(tesseract, cube))

    val p = applied.head.projection4D.get
    (p.rotXW, p.rotYW, p.rotZW) shouldBe (15f, 17f, 2f)
    applied(1) shouldBe cube

  it should "leave the specs alone for a zero offset" in:
    Interactive4DOffset.Zero.applyTo(List(tesseract)) shouldBe List(tesseract)

  it should "accumulate rotations" in:
    val offset = Interactive4DOffset.Zero.plus(1f, 0f, 0f).plus(2f, 4f, 0f)

    (offset.rotXW, offset.rotYW) shouldBe (3f, 4f)
    offset.isZero shouldBe false

  "Interactive4DOffset.fileProjections" should "differ when the file changed a projection" in:
    val changed = tesseract.copy(projection4D = Some(Projection4DSpec(rotXW = 99f)))

    Interactive4DOffset.fileProjections(List(tesseract)) shouldBe
      Interactive4DOffset.fileProjections(List(tesseract.copy(x = 1f)))
    Interactive4DOffset.fileProjections(List(tesseract)) should not be
      Interactive4DOffset.fileProjections(List(changed))

class WindowStateFileSuite extends AnyFlatSpec with Matchers:

  private val offset = Interactive4DOffset().plus(5f, 0f, 0f)
  private def state(rotXW: Float = 5f) = WindowStateFile.of(
    Vector[3](0f, 1f, 6f), Vector[3](0f, 0f, 0f), Vector[3](0f, 1f, 0f),
    Interactive4DOffset().plus(rotXW, 0f, 0f), Some(1.5f), Some(true)
  )

  private def target = new File(Files.createTempDirectory("window-state-").toFile, "s.scala.window-state.json")

  "WindowStateFile.pathFor" should "sit next to the scene file" in:
    WindowStateFile.pathFor(new File("/x/current.scala")).getPath shouldBe
      "/x/current.scala.window-state.json"

  "WindowStateFile" should "write the camera, the 4D rotation and the playback state" in:
    val file = target
    WindowStateFile(file, () => 0L).update(state())

    val text = Files.readString(file.toPath)
    text should include("\"rotXW\": 5")
    text should include("\"cameraEye\"")
    text should include("\"playing\": true")

  it should "write at most every 250 ms, and only a changed state" in:
    val file = target
    val time = new java.util.concurrent.atomic.AtomicLong(0L)
    val writer = WindowStateFile(file, () => time.get())
    writer.update(state(5f))
    time.set(100L)
    writer.update(state(9f)) // too soon: not written yet
    Files.readString(file.toPath) should include("\"rotXW\": 5")
    time.set(300L)
    writer.update(state(9f))
    Files.readString(file.toPath) should include("\"rotXW\": 9")
    file.delete()
    time.set(900L)
    writer.update(state(9f)) // unchanged: no rewrite
    file.exists() shouldBe false
