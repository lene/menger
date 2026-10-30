package menger.engines

import java.io.File
import java.nio.file.Files

import menger.common.Vector
import menger.spec.ObjectSpec
import menger.spec.Projection4DSpec
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

  // Session 5, F116 (menger#131): the preview window ignored Shift+scroll.
  it should "move eye-w by one power of 1.1 per Shift+scroll notch, as the interactive window" in:
    val up = menger.RotationProjectionParameters(0, 0, 0, eyeW = 65f)
    val down = menger.RotationProjectionParameters(0, 0, 0, eyeW = 1.0156f)
    val eyeW = (o: Interactive4DOffset) => o.applyTo(List(tesseract)).head.projection4D.get.eyeW
    val fileEyeW = tesseract.projection4D.get.eyeW.toDouble
    eyeW(Interactive4DOffset.Zero.plus(up)) shouldBe math.pow(fileEyeW, 1.1).toFloat +- 1e-5f
    Interactive4DOffset.Zero.plus(up).plus(down).isZero shouldBe true
    // A rotation-only event carries the default eye-w: no notch.
    Interactive4DOffset.Zero.plus(menger.RotationProjectionParameters(rotXW = 5f)).eyeWSteps
      .shouldBe(0)
    // Never at or below screen-w, which Projection4DSpec refuses.
    val far = (1 to 60).foldLeft(Interactive4DOffset.Zero)((o, _) => o.plus(down))
    eyeW(far) should be > tesseract.projection4D.get.screenW

  "Window4DState.describe" should "show the first 4D object's rotation and eye-w with the controls" in:
    val state = Window4DState.describe(List(cube, tesseract)).get
    state should startWith("4D xw 10.0° yw 20.0° zw 0.0° eye-w 3.00")
    state should include("Shift+drag")
    Window4DState.describe(List(cube)) shouldBe None

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
