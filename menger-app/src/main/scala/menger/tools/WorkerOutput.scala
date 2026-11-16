package menger.tools

import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.nio.charset.StandardCharsets.UTF_8

import scala.annotation.tailrec

/** What the parent validator reads from its worker's standard output (menger#137): the worker's
  * result line, found by its nonce (see [[WorkerProtocol]]), and a bounded sample of everything
  * else, which is the scene's output.
  *
  * Nothing here grows with the amount the scene prints: the scene's lines are read and thrown
  * away, a line is kept only up to [[MaxQuotedLineBytes]], and only [[MaxQuotedLines]] of them. */
private[tools] object WorkerOutput:

  /** A scene's lines kept for the message of a refusal, first lines first. */
  val MaxQuotedLines = 20
  val MaxQuotedLineBytes = 200

  /** The longest result line accepted: a real result is a few hundred kilobytes at most, and the
    * parent runs with a small heap. */
  val MaxResultBytes = 4 * 1024 * 1024

  /** `result` is the text after the nonce on the first line that carried it (None if no line
    * did); `resultTooLarge` is set when that line was longer than [[MaxResultBytes]], and then
    * `result` is None as well. `sceneOutput` is the first [[MaxQuotedLines]] lines of the scene,
    * each cut to [[MaxQuotedLineBytes]]; `sceneLines` and `sceneBytes` count all of them. */
  final case class Captured(
    result: Option[String],
    resultTooLarge: Boolean,
    sceneOutput: List[String],
    sceneLines: Long,
    sceneBytes: Long
  )

  private val EndOfStream = -1
  private val LineFeed = '\n'.toInt

  private final case class RawLine(kept: Array[Byte], length: Long)

  /** Reads `source` until the result line or the end of the stream. Stops reading at the result
    * line: the worker has nothing more to say, and a scene could go on printing forever. */
  def read(source: InputStream, nonce: String): Captured =
    val in = new BufferedInputStream(source)
    val prefix = WorkerProtocol.resultPrefix(nonce).getBytes(UTF_8)

    @tailrec def loop(acc: Captured): Captured =
      in.mark(prefix.length)
      val head = in.readNBytes(prefix.length)
      in.reset()
      if head.sameElements(prefix) then
        val line = readLine(in, MaxResultBytes)
        line.fold(acc)(resultFrom(acc, _, prefix.length))
      else readLine(in, MaxQuotedLineBytes) match
        case None       => acc
        case Some(line) => loop(withSceneLine(acc, line))

    loop(Captured(None, false, Nil, 0L, 0L))

  private def resultFrom(acc: Captured, line: RawLine, prefixLength: Int): Captured =
    if line.length > MaxResultBytes then acc.copy(resultTooLarge = true)
    else acc.copy(result = Some(new String(line.kept, prefixLength, line.kept.length - prefixLength, UTF_8)))

  private def withSceneLine(acc: Captured, line: RawLine): Captured =
    val quoted =
      if acc.sceneOutput.size < MaxQuotedLines then
        acc.sceneOutput :+ new String(line.kept, UTF_8)
      else acc.sceneOutput
    acc.copy(sceneOutput = quoted, sceneLines = acc.sceneLines + 1, sceneBytes = acc.sceneBytes + line.length)

  /** The next line without its line break, or None at the end of the stream. Keeps at most `keep`
    * bytes of it and counts the rest. */
  private def readLine(in: InputStream, keep: Int): Option[RawLine] =
    val kept = new ByteArrayOutputStream()

    @tailrec def next(length: Long): Option[RawLine] = in.read() match
      case EndOfStream if length == 0L => None
      case EndOfStream | LineFeed      => Some(RawLine(kept.toByteArray, length))
      case byte =>
        if length < keep then kept.write(byte)
        next(length + 1)

    next(0L)
