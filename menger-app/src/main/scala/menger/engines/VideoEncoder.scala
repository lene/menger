package menger.engines

import scala.sys.process._
import scala.util.Try

import com.typesafe.scalalogging.LazyLogging

object VideoEncoder extends LazyLogging:

  val DefaultQp = 12

  private enum Codec(val name: String, val extraArgs: List[String]):
    case H264 extends Codec("libx264",    List("-pix_fmt", "yuv420p"))
    case Hevc extends Codec("hevc_nvenc", Nil)

  private def codecForPath(outputPath: String): Codec =
    val lower = outputPath.toLowerCase
    if lower.endsWith(".mp4") then Codec.H264
    else if lower.endsWith(".mkv") then Codec.Hevc
    else sys.error(
      s"Unsupported video format for '$outputPath'. Supported extensions: .mp4 (H.264/libx264), .mkv (HEVC/hevc_nvenc)"
    )

  /** Build the ffmpeg argument list. Separated from encode() for testability. */
  def buildArgs(inputPattern: String, outputPath: String, fps: Int, quality: Int): List[String] =
    val codec = codecForPath(outputPath)
    List("ffmpeg", "-y", "-framerate", fps.toString, "-i", inputPattern) ++
      List("-c:v", codec.name, "-qp", quality.toString) ++
      codec.extraArgs ++
      List(outputPath)

  /** Verify ffmpeg is on PATH and the encoder for outputPath's format is available.
   *  Throws (via sys.error) if either check fails.
   *  Safe to call at application startup.
   */
  def checkAvailable(outputPath: String): Unit =
    val ffmpegFound = Try("ffmpeg -version" !! ProcessLogger(_ => ())).isSuccess
    if !ffmpegFound then
      sys.error("ffmpeg not found on PATH. Install ffmpeg to use --video output.")
    val codec = codecForPath(outputPath)
    val encoderPresent =
      Try("ffmpeg -encoders" !! ProcessLogger(_ => ())).toOption
        .exists(_.contains(codec.name))
    if !encoderPresent then
      sys.error(
        s"ffmpeg encoder '${codec.name}' is not available on this system. " +
        "Install the required codec or choose a different output format."
      )

  /** Encode frame sequence into a video file.
   *
   *  @param inputPattern  Printf-style frame pattern, e.g. "frames/frame_%04d.png"
   *  @param outputPath    Output video path; extension determines codec (.mp4 or .mkv)
   *  @param fps           Frames per second (default 24)
   *  @param quality       QP value (0=lossless, 51=worst; default 12 = master quality)
   */
  def encode(
    inputPattern: String,
    outputPath: String,
    fps: Int,
    quality: Int = DefaultQp
  ): Unit =
    val args = buildArgs(inputPattern, outputPath, fps, quality)
    logger.info(s"Encoding video: ${args.mkString(" ")}")
    val stderr = new StringBuilder
    val exitCode = Process(args).!(ProcessLogger(_ => (), line => { val _ = stderr.append(line).append('\n') }))
    if exitCode != 0 then
      logger.error(s"ffmpeg stderr:\n$stderr")
      sys.error(
        s"ffmpeg exited with code $exitCode encoding '$outputPath'. See log for details."
      )
    logger.info(s"Video written to $outputPath")
