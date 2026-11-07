package menger.tools

import java.io.ByteArrayOutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

import scala.util.Using

import org.scalatest.BeforeAndAfterAll
import org.scalatest.EitherValues
import org.scalatest.OptionValues
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class HdriSunSuite extends AnyFlatSpec with Matchers with EitherValues with OptionValues
    with BeforeAndAfterAll:

  private val Width = 256
  private val Height = 128
  private val Background = 0.01
  private val SunValue = 100.0

  private val tempDir: Path = Files.createTempDirectory("hdri-sun-suite")

  override def afterAll(): Unit =
    Using.resource(Files.list(tempDir))(_.forEach(Files.delete(_)))
    Files.delete(tempDir)
    super.afterAll()

  private type Pixels = (Int, Int) => (Double, Double, Double)

  // ---- a minimal Radiance writer: flat or new-style RLE scanlines ---------------------

  private def rgbeOf(rgb: (Double, Double, Double)): Array[Byte] =
    val largest = rgb._1.max(rgb._2).max(rgb._3)
    if largest < 1e-32 then Array.fill(4)(0.toByte)
    else
      val exponent = Math.getExponent(largest) + 1
      val scale = Math.scalb(1.0, 8 - exponent)
      Array(rgb._1, rgb._2, rgb._3).map(c => (c * scale).toInt.toByte) :+ (exponent + 128).toByte

  private def rleChannel(values: Array[Byte]): Array[Byte] =
    val out = new ByteArrayOutputStream()
    def runLength(from: Int): Int =
      values.drop(from).takeWhile(_ == values(from)).length.min(127)
    def encode(from: Int): Unit =
      if from < values.length then
        val run = runLength(from)
        if run >= 3 then
          out.write(128 + run)
          out.write(values(from))
          encode(from + run)
        else
          val literal = values.length.min(from + 3) - from
          out.write(literal)
          out.write(values, from, literal)
          encode(from + literal)
    encode(0)
    out.toByteArray

  private def scanline(row: Int, width: Int, pixels: Pixels, rle: Boolean): Array[Byte] =
    val rgbe = (0 until width).map(x => rgbeOf(pixels(x, row)))
    if rle then
      Array[Byte](2, 2, (width >> 8).toByte, width.toByte) ++
        (0 until 4).flatMap(c => rleChannel(rgbe.map(_(c)).toArray)).toArray
    else rgbe.flatten.toArray

  private def hdrBytes(
    width: Int, height: Int, pixels: Pixels, rle: Boolean = true,
    resolution: String = ""
  ): Array[Byte] =
    val resolutionLine = if resolution.nonEmpty then resolution else s"-Y $height +X $width"
    val header = s"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n$resolutionLine\n"
    header.getBytes(StandardCharsets.ISO_8859_1) ++
      (0 until height).flatMap(row => scanline(row, width, pixels, rle)).toArray

  private def writeHdr(name: String, bytes: Array[Byte]): Path =
    Files.write(tempDir.resolve(name), bytes)

  private def spotPixels(columns: Set[Int], rows: Set[Int], color: (Double, Double, Double)) =
    (x: Int, y: Int) =>
      if columns.contains(x) && rows.contains(y) then color
      else (Background, Background, Background)

  private val white = (SunValue, SunValue, SunValue)

  private def sunOf(bytes: Array[Byte]): HdriSun.SunEstimate =
    HdriSun.decode(bytes).flatMap(HdriSun.findSun).value

  // ---- the matrix ---------------------------------------------------------------------

  "HdriSun" should "map a bright spot back to its (u, v) within half a pixel" in:
    val sun = sunOf(hdrBytes(Width, Height, spotPixels(Set(99, 100, 101), Set(39, 40, 41), white)))
    (sun.u * Width) shouldBe (100.5 +- 0.5)
    (sun.v * Height) shouldBe (40.5 +- 0.5)
    val (u, v) = HdriSun.uvOf(sun.towardSun)
    (u, v) shouldBe ((sun.u, sun.v))

  it should "use the background's mapping: +Y up, atan2(z, x) for the azimuth" in:
    HdriSun.directionOf(0.5, 0.0).y shouldBe 1.0 +- 1e-9
    val ahead = HdriSun.directionOf(0.5, 0.5)
    ahead.x shouldBe 1.0 +- 1e-9
    ahead.z shouldBe 0.0 +- 1e-9
    HdriSun.directionOf(0.75, 0.5).z shouldBe 1.0 +- 1e-9

  it should "read flat scanlines the same as run-length encoded ones" in:
    val pixels = spotPixels(Set(3, 4), Set(2, 3), white)
    val flat = sunOf(hdrBytes(Width, Height, pixels, rle = false))
    val encoded = sunOf(hdrBytes(Width, Height, pixels, rle = true))
    flat.towardSun shouldBe encoded.towardSun
    flat.pixels shouldBe encoded.pixels

  it should "find one sun for a spot split across the u seam" in:
    val sun = sunOf(hdrBytes(Width, Height, spotPixels(Set(Width - 1, 0), Set(39, 40, 41), white)))
    sun.pixels shouldBe 6
    val seamDistance = math.min(sun.u, 1.0 - sun.u) * Width
    seamDistance shouldBe (0.0 +- 0.5)
    sun.towardSun.x shouldBe (-math.cos(math.toRadians(sun.elevationDegrees)) +- 0.001)
    sun.towardSun.z shouldBe (0.0 +- 0.01)

  it should "report the angular diameter of the region" in:
    val (width, height) = (2048, 1024)
    val block = (1022 to 1025).toSet
    val sun = sunOf(hdrBytes(width, height, spotPixels(block, (510 to 513).toSet, white)))
    val solidAngle = 16 * (math.Pi / height) * (2 * math.Pi / width)
    val expected = math.toDegrees(2 * math.sqrt(solidAngle / math.Pi))
    sun.diameterDegrees shouldBe (expected +- (0.02 * expected))
    sun.elevationDegrees shouldBe (0.0 +- 0.1)

  it should "report the region's colour with the largest channel at 1" in:
    val orange = (SunValue, 0.86 * SunValue, 0.4 * SunValue)
    val sun = sunOf(hdrBytes(Width, Height, spotPixels(Set(99, 100, 101), Set(39, 40, 41), orange)))
    sun.color._1 shouldBe (1.0 +- 0.02)
    sun.color._2 shouldBe (0.86 +- 0.02)
    sun.color._3 shouldBe (0.4 +- 0.02)

  it should "report the sun's share of the sky's light" in:
    val sun = sunOf(hdrBytes(Width, Height, spotPixels(Set(100, 101), Set(40, 41), white)))
    sun.skyShare should (be > 0.0 and be < 1.0)
    val dark = sunOf(hdrBytes(Width, Height,
      (x, y) => if x == 100 && y == 40 then white else (0.0, 0.0, 0.0)))
    dark.skyShare shouldBe (1.0 +- 1e-6)

  it should "ignore a bright region that is not connected to the peak" in:
    val pixels = (x: Int, y: Int) =>
      if x == 100 && y == 40 then (SunValue, SunValue, SunValue)
      else if x == 200 && y == 90 then (0.5 * SunValue, 0.5 * SunValue, 0.5 * SunValue)
      else (Background, Background, Background)
    val sun = sunOf(hdrBytes(Width, Height, pixels))
    sun.pixels shouldBe 1
    (sun.u * Width) shouldBe (100.5 +- 0.5)

  it should "print a Sun line pointing the light away from the sun" in:
    val sun = sunOf(hdrBytes(Width, Height, spotPixels(Set(99, 100, 101), Set(39, 40, 41), white)))
    val line = HdriSun.sunLine(sun)
    line should startWith("Sun(Vec3(")
    line should include("diameterDegrees = ")
    line should include("color = Color(1.00f")
    val numbers = """Vec3\((-?\d+\.\d+)f, (-?\d+\.\d+)f, (-?\d+\.\d+)f\)""".r
      .findFirstMatchIn(line).map(_.subgroups.map(_.toDouble)).value
    numbers.zip(List(sun.towardSun.x, sun.towardSun.y, sun.towardSun.z)).foreach {
      case (printed, toward) => printed shouldBe (-toward +- 1e-4)
    }

  it should "refuse a flat image, naming the file" in:
    val path = writeHdr("flat.hdr", hdrBytes(Width, Height, (_, _) => (1.0, 1.0, 1.0)))
    val message = HdriSun.run(Array(path.toString)).left.value
    message should include(path.toString)
    message should include("no sun found")

  it should "refuse a flat 4K map without exhausting memory" in:
    val (width, height) = (4096, 2048)
    val header = s"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y $height +X $width\n"
    val pixel = rgbeOf((1.0, 1.0, 1.0))
    val body = Array.tabulate(width * height * 4)(i => pixel(i % 4))
    val path = writeHdr("flat4k.hdr", header.getBytes(StandardCharsets.ISO_8859_1) ++ body)
    val message = HdriSun.run(Array(path.toString)).left.value
    message should include("no sun found")
    message should include("wider than 10 degrees")

  it should "refuse a black image" in:
    val path = writeHdr("black.hdr", hdrBytes(Width, Height, (_, _) => (0.0, 0.0, 0.0)))
    HdriSun.run(Array(path.toString)).left.value should include("no sun found")

  private val spot = spotPixels(Set(100), Set(40), white)

  private def refusalOf(name: String, bytes: Array[Byte]): String =
    val path = writeHdr(name, bytes)
    val message = HdriSun.run(Array(path.toString)).left.value
    message should include(path.toString)
    message

  it should "refuse a truncated file" in:
    val bytes = hdrBytes(Width, Height, spot)
    refusalOf("truncated.hdr", bytes.take(bytes.length / 2)) should include("truncated")

  it should "refuse a file with the wrong magic" in:
    val bytes = hdrBytes(Width, Height, spot)
    refusalOf("magic.hdr", "P6".getBytes(StandardCharsets.ISO_8859_1) ++ bytes.drop(2)) should
      include("not a Radiance file")

  it should "refuse a layout other than -Y height +X width" in:
    List("+Y 128 +X 256", "-Y 128 -X 256", "-X 256 -Y 128").foreach { resolution =>
      withClue(resolution) {
        refusalOf("layout.hdr", hdrBytes(Width, Height, spot, resolution = resolution)) should
          include("unsupported layout")
      }
    }

  it should "refuse a non-positive size" in:
    List("-Y 0 +X 256", "-Y 128 +X 0", "-Y -128 +X 256").foreach { resolution =>
      withClue(resolution) {
        refusalOf("size.hdr", hdrBytes(Width, Height, spot, resolution = resolution)) should
          include("size must be positive")
      }
    }

  it should "refuse a format other than RGBE" in:
    val bytes = hdrBytes(Width, Height, spot)
    val xyze = new String(bytes, StandardCharsets.ISO_8859_1)
      .replace("32-bit_rle_rgbe", "32-bit_rle_xyze").getBytes(StandardCharsets.ISO_8859_1)
    refusalOf("xyze.hdr", xyze) should include("unsupported FORMAT")

  it should "refuse a missing file, naming the path" in:
    val missing = tempDir.resolve("does-not-exist.hdr").toString
    HdriSun.run(Array(missing)).left.value should include(missing)

  it should "print a usage message without arguments" in:
    HdriSun.run(Array.empty).left.value should startWith("Usage")

  // ---- Kiara 2 Sunrise (menger-toplevel#98 G3): fetched, not committed ---------------------

  private val kiara4k = List("assets/hdri", "../assets/hdri")
    .map(dir => Path.of(dir, "kiara_2_sunrise_4k.hdr")).find(Files.isRegularFile(_))

  it should "find the Kiara 4K sun where the background shows it" in:
    val path = kiara4k.getOrElse(
      cancel("assets/hdri/kiara_2_sunrise_4k.hdr is not fetched (menger#157)"))
    val sun = HdriSun.run(Array(path.toString)).value
    info(sun)
    val estimate = HdriSun.decode(Files.readAllBytes(path)).flatMap(HdriSun.findSun).value
    estimate.towardSun.x shouldBe (0.742 +- 0.005)
    estimate.towardSun.y shouldBe (0.072 +- 0.005)
    estimate.towardSun.z shouldBe (0.666 +- 0.005)
    estimate.elevationDegrees shouldBe (4.1 +- 0.2)
    estimate.diameterDegrees shouldBe (0.5 +- 0.1)
    estimate.color._1 shouldBe (1.0 +- 0.03)
    estimate.color._2 shouldBe (0.86 +- 0.03)
    estimate.color._3 shouldBe (0.40 +- 0.03)
