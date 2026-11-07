package menger.tools

import java.nio.BufferUnderflowException
import java.nio.ByteBuffer
import java.nio.file.Files
import java.nio.file.Paths
import java.util.Locale

import scala.annotation.tailrec
import scala.collection.mutable
import scala.util.Try

import com.typesafe.scalalogging.LazyLogging

/** Finds the sun in an equirectangular Radiance `.hdr` and prints the values of a matching DSL
  * `Sun` light (menger#158). The scene's caustics come only from scene lights -- the environment
  * map emits no photons -- so a key light has to sit where the sun shows in the sky.
  *
  * The method, in one place:
  *  - directions use the background's mapping (`miss_plane.cu`): u = 0.5 + atan2(z, x) / 2pi,
  *    v = 0.5 - asin(y) / pi, +Y up, top row first;
  *  - the sun is the region of pixels at 10 % of the peak luminance or more that is connected to
  *    the peak (8-neighbourhood, wrapping at the u seam);
  *  - its direction is the luminance x solid-angle weighted mean of the pixels' directions, its
  *    diameter that of the spherical cap with the region's solid angle, its colour the region's
  *    solid-angle weighted mean RGB scaled to a largest channel of 1;
  *  - its share of the sky's light is its luminance x solid angle over the whole sphere's. */
object HdriSun extends LazyLogging:

  val PeakFraction = 0.1
  val MaxSunDiameterDegrees: Double = menger.dsl.Sun.MaxDiameterDegrees.toDouble
  val MaxPixels: Long = 1L << 26

  private val RadianceMagicPrefix = "#?"
  private val SupportedFormat = "32-bit_rle_rgbe"
  private val LayoutPattern = """([+-][XY])\s+(-?\d+)\s+([+-][XY])\s+(-?\d+)""".r
  private val SupportedLayout = ("-Y", "+X")
  private val MinRleWidth = 8
  private val MaxRleWidth = 32767
  private val RleMarker = 2
  private val RleRunFlag = 128
  private val RgbeChannels = 4
  private val ExponentBias = 136
  private val LuminanceWeights = (0.2126, 0.7152, 0.0722)
  private val PercentPerUnit = 100.0

  final case class Direction(x: Double, y: Double, z: Double):
    def length: Double = math.sqrt(x * x + y * y + z * z)
    def normalized: Direction = Direction(x / length, y / length, z / length)
    def negated: Direction = Direction(-x, -y, -z)

  final case class HdrImage(width: Int, height: Int, rgbe: Array[Byte]):
    def pixelCount: Int = width * height

    def rgb(index: Int): (Double, Double, Double) =
      val base = index * RgbeChannels
      val exponent = rgbe(base + 3) & 0xFF
      if exponent == 0 then (0.0, 0.0, 0.0)
      else
        val scale = Math.scalb(1.0, exponent - ExponentBias)
        def channel(offset: Int): Double = (rgbe(base + offset) & 0xFF) * scale
        (channel(0), channel(1), channel(2))

  final case class SunEstimate(
    towardSun: Direction,
    u: Double,
    v: Double,
    elevationDegrees: Double,
    diameterDegrees: Double,
    color: (Double, Double, Double),
    skyShare: Double,
    pixels: Int
  ):
    def travelDirection: Direction = towardSun.negated

  def luminance(rgb: (Double, Double, Double)): Double =
    LuminanceWeights._1 * rgb._1 + LuminanceWeights._2 * rgb._2 + LuminanceWeights._3 * rgb._3

  /** The background's mapping from a direction to the map's (u, v). */
  def uvOf(direction: Direction): (Double, Double) =
    val d = direction.normalized
    (0.5 + math.atan2(d.z, d.x) / (2 * math.Pi), 0.5 - math.asin(d.y.max(-1.0).min(1.0)) / math.Pi)

  /** The inverse of `uvOf`. */
  def directionOf(u: Double, v: Double): Direction =
    val elevation = (0.5 - v) * math.Pi
    val azimuth = (u - 0.5) * 2 * math.Pi
    Direction(
      math.cos(elevation) * math.cos(azimuth),
      math.sin(elevation),
      math.cos(elevation) * math.sin(azimuth)
    )

  // ---- reading -------------------------------------------------------------------------

  def decode(bytes: Array[Byte]): Either[String, HdrImage] =
    Try(decodeUnchecked(ByteBuffer.wrap(bytes))).fold(
      {
        case _: BufferUnderflowException => Left("the file is truncated")
        case other                       => Left(s"cannot decode: ${other.getMessage}")
      },
      identity
    )

  private def decodeUnchecked(buffer: ByteBuffer): Either[String, HdrImage] =
    val magic = readLine(buffer)
    if !magic.startsWith(RadianceMagicPrefix) then
      Left("not a Radiance file: the \"#?RADIANCE\" magic line is missing")
    else
      for
        _ <- checkFormat(readHeader(buffer))
        size <- parseLayout(readLine(buffer))
        image <- readScanlines(buffer, size._1, size._2)
      yield image

  @tailrec
  private def readHeader(buffer: ByteBuffer, lines: List[String] = Nil): List[String] =
    val line = readLine(buffer)
    if line.isEmpty then lines.reverse else readHeader(buffer, line :: lines)

  private def readLine(buffer: ByteBuffer): String =
    @tailrec def collect(chars: StringBuilder): String =
      val byte = buffer.get()
      if byte == '\n'.toByte then chars.toString else collect(chars.append((byte & 0xFF).toChar))
    collect(new StringBuilder)

  private def checkFormat(header: List[String]): Either[String, Unit] =
    header.find(_.startsWith("FORMAT=")).map(_.stripPrefix("FORMAT=")) match
      case Some(format) if format != SupportedFormat =>
        Left(s"unsupported FORMAT=$format (only $SupportedFormat)")
      case _ => Right(())

  private def parseLayout(line: String): Either[String, (Int, Int)] =
    line.trim match
      case LayoutPattern(first, height, second, width) if (first, second) == SupportedLayout =>
        val size = (height.toLongOption, width.toLongOption)
        size match
          case (Some(h), Some(w)) if h > 0 && w > 0 && h * w <= MaxPixels =>
            Right((w.toInt, h.toInt))
          case (Some(h), Some(w)) if h > 0 && w > 0 =>
            Left(s"the image is too large (${w}x$h, at most $MaxPixels pixels)")
          case _ => Left(s"the image size must be positive, got \"${line.trim}\"")
      case LayoutPattern(first, _, second, _) =>
        Left(s"unsupported layout \"$first .. $second\" (only \"-Y height +X width\")")
      case other => Left(s"unreadable resolution line \"$other\"")

  private def readScanlines(buffer: ByteBuffer, width: Int, height: Int): Either[String, HdrImage] =
    val rgbe = new Array[Byte](width * height * RgbeChannels)
    @tailrec def scanlinesFrom(row: Int): Either[String, HdrImage] =
      if row >= height then Right(HdrImage(width, height, rgbe))
      else
        readScanline(buffer, width, rgbe, row * width * RgbeChannels) match
          case Left(problem) => Left(s"scanline $row: $problem")
          case Right(_)      => scanlinesFrom(row + 1)
    scanlinesFrom(0)

  private def readScanline(
    buffer: ByteBuffer, width: Int, out: Array[Byte], offset: Int
  ): Either[String, Unit] =
    if isRleScanline(buffer, width) then readRleScanline(buffer, width, out, offset)
    else
      buffer.get(out, offset, width * RgbeChannels)
      Right(())

  private def isRleScanline(buffer: ByteBuffer, width: Int): Boolean =
    val at = buffer.position()
    width >= MinRleWidth && width <= MaxRleWidth && buffer.remaining() >= RgbeChannels &&
      (buffer.get(at) & 0xFF) == RleMarker && (buffer.get(at + 1) & 0xFF) == RleMarker &&
      (buffer.get(at + 2) & 0x80) == 0

  private def readRleScanline(
    buffer: ByteBuffer, width: Int, out: Array[Byte], offset: Int
  ): Either[String, Unit] =
    val declaredWidth = ((buffer.get(buffer.position() + 2) & 0xFF) << 8) |
      (buffer.get(buffer.position() + 3) & 0xFF)
    buffer.position(buffer.position() + RgbeChannels)
    if declaredWidth != width then
      Left(s"the scanline declares width $declaredWidth, the image has $width")
    else
      val planes = new Array[Byte](width * RgbeChannels)
      @tailrec def channelsFrom(channel: Int): Either[String, Unit] =
        if channel >= RgbeChannels then Right(())
        else
          readRleChannel(buffer, planes, channel * width, width) match
            case Left(problem) => Left(problem)
            case Right(_)      => channelsFrom(channel + 1)
      channelsFrom(0).map { _ =>
        (0 until width).foreach { x =>
          (0 until RgbeChannels).foreach { c =>
            out(offset + x * RgbeChannels + c) = planes(c * width + x)
          }
        }
      }

  @tailrec
  private def readRleChannel(
    buffer: ByteBuffer, planes: Array[Byte], start: Int, width: Int, filled: Int = 0
  ): Either[String, Unit] =
    if filled >= width then Right(())
    else
      val code = buffer.get() & 0xFF
      val isRun = code > RleRunFlag
      val length = if isRun then code - RleRunFlag else code
      if length == 0 || filled + length > width then Left("a run does not fit its scanline")
      else
        if isRun then
          java.util.Arrays.fill(planes, start + filled, start + filled + length, buffer.get())
        else buffer.get(planes, start + filled, length)
        readRleChannel(buffer, planes, start, width, filled + length)

  // ---- finding the sun -----------------------------------------------------------------

  def findSun(image: HdrImage): Either[String, SunEstimate] =
    val luminances = Array.tabulate(image.pixelCount)(i => luminance(image.rgb(i)))
    val peak = luminances.max
    if peak <= 0.0 then Left("no sun found: the image is black")
    else
      val solidAngles = Array.tabulate(image.height)(rowSolidAngle(image, _))
      val peakIndex = luminances.indexOf(peak)
      val members = connectedToPeak(image, luminances, peakIndex, PeakFraction * peak)
      estimate(image, luminances, solidAngles, members)

  private def rowSolidAngle(image: HdrImage, row: Int): Double =
    val elevation = (0.5 - (row + 0.5) / image.height) * math.Pi
    math.cos(elevation) * (math.Pi / image.height) * (2 * math.Pi / image.width)

  private def connectedToPeak(
    image: HdrImage, luminances: Array[Double], peakIndex: Int, threshold: Double
  ): Array[Int] =
    val isMember = new Array[Boolean](image.pixelCount)
    val members = mutable.ArrayBuffer(peakIndex)
    val pending = mutable.ArrayDeque(peakIndex)
    isMember(peakIndex) = true
    @tailrec def drain(): Unit = pending.removeLastOption() match
      case None => ()
      case Some(index) =>
        neighbours(image, index).filter(n => !isMember(n) && luminances(n) >= threshold)
          .foreach { n =>
            isMember(n) = true
            members += n
            pending.append(n)
          }
        drain()
    drain()
    members.toArray

  private def neighbours(image: HdrImage, index: Int): Seq[Int] =
    val (row, column) = (index / image.width, index % image.width)
    for
      dRow <- -1 to 1
      dColumn <- -1 to 1
      if (dRow, dColumn) != (0, 0)
      if row + dRow >= 0 && row + dRow < image.height
    yield (row + dRow) * image.width + (column + dColumn + image.width) % image.width

  private def estimate(
    image: HdrImage, luminances: Array[Double], solidAngles: Array[Double],
    members: Array[Int]
  ): Either[String, SunEstimate] =
    def solidAngleOf(index: Int): Double = solidAngles(index / image.width)
    def directionAt(index: Int): Direction =
      directionOf(
        ((index % image.width) + 0.5) / image.width,
        ((index / image.width) + 0.5) / image.height
      )
    val regionSolidAngle = members.map(solidAngleOf).sum
    val capHalfAngle = math.acos((1 - regionSolidAngle / (2 * math.Pi)).max(-1.0))
    val diameterDegrees = math.toDegrees(2 * capHalfAngle)
    val weighted = members.map { i =>
      val (d, w) = (directionAt(i), luminances(i) * solidAngleOf(i))
      (d.x * w, d.y * w, d.z * w)
    }
    val sum = Direction(weighted.map(_._1).sum, weighted.map(_._2).sum, weighted.map(_._3).sum)
    if diameterDegrees > MaxSunDiameterDegrees || sum.length == 0.0 then
      Left(
        "no sun found: the brightest region (at least " +
          s"${fixed(PeakFraction * PercentPerUnit, 0)} % of the peak) is " +
          s"${fixed(diameterDegrees, 1)} degrees wide, a sun is at most " +
          s"${fixed(MaxSunDiameterDegrees, 0)}"
      )
    else
      val toward = sum.normalized
      val (u, v) = uvOf(toward)
      val regionLight = members.map(i => luminances(i) * solidAngleOf(i)).sum
      Right(SunEstimate(
        toward, u, v, math.toDegrees(math.asin(toward.y)), diameterDegrees,
        meanColor(image, members, solidAngleOf),
        regionLight / totalLight(image, luminances, solidAngles), members.length
      ))

  private def totalLight(
    image: HdrImage, luminances: Array[Double], solidAngles: Array[Double]
  ): Double =
    (0 until image.height).map { row =>
      solidAngles(row) *
        java.util.Arrays.stream(luminances, row * image.width, (row + 1) * image.width).sum()
    }.sum

  private def meanColor(
    image: HdrImage, members: Array[Int], solidAngleOf: Int => Double
  ): (Double, Double, Double) =
    val channels = members.map(i => image.rgb(i)).zip(members.map(solidAngleOf))
      .map { case ((r, g, b), omega) => (r * omega, g * omega, b * omega) }
    val (r, g, b) = (channels.map(_._1).sum, channels.map(_._2).sum, channels.map(_._3).sum)
    val largest = r.max(g).max(b)
    (r / largest, g / largest, b / largest)

  // ---- reporting -----------------------------------------------------------------------

  private def fixed(value: Double, digits: Int): String =
    String.format(Locale.ROOT, s"%.${digits}f", java.lang.Double.valueOf(value))

  private def triple(values: (Double, Double, Double), digits: Int, suffix: String = ""): String =
    List(values._1, values._2, values._3).map(fixed(_, digits) + suffix).mkString(", ")

  def sunLine(sun: SunEstimate): String =
    val d = sun.travelDirection
    s"Sun(Vec3(${triple((d.x, d.y, d.z), 4, "f")}), " +
      s"diameterDegrees = ${fixed(sun.diameterDegrees, 2)}f, " +
      s"color = Color(${triple(sun.color, 2, "f")}))"

  def report(fileName: String, image: HdrImage, sun: SunEstimate): String =
    val t = sun.towardSun
    List(
      s"file: $fileName (${image.width}x${image.height})",
      s"direction to the sun (background convention): (${triple((t.x, t.y, t.z), 3)})",
      s"position in the map: u = ${fixed(sun.u, 4)}, v = ${fixed(sun.v, 4)}",
      s"elevation: ${fixed(sun.elevationDegrees, 1)} degrees",
      s"diameter: ${fixed(sun.diameterDegrees, 2)} degrees (${sun.pixels} pixels at " +
        s"${fixed(PeakFraction * PercentPerUnit, 0)} % of the peak luminance or more)",
      s"colour: (${triple(sun.color, 3)})",
      s"share of the sky's light: ${fixed(sun.skyShare * PercentPerUnit, 0)} % (an IBL scene " +
        "counts it twice, once as sky and once as the Sun light)",
      sunLine(sun)
    ).mkString(System.lineSeparator())

  def run(args: Array[String]): Either[String, String] =
    args.toList match
      case List(path) =>
        for
          bytes <- Try(Files.readAllBytes(Paths.get(path))).toEither.left.map(_ =>
            s"cannot read $path: no such readable file")
          image <- decode(bytes).left.map(problem => s"$path: $problem")
          sun <- findSun(image).left.map(problem => s"$path: $problem")
        yield report(path, image, sun)
      case _ => Left("Usage: HdriSun <file.hdr>")

  def main(args: Array[String]): Unit =
    run(args) match
      case Right(text) =>
        print(text)
        print(System.lineSeparator())
      case Left(message) =>
        logger.error(message)
        sys.exit(1)
