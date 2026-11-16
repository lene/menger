package menger.tools

import java.nio.charset.StandardCharsets.UTF_8
import java.security.SecureRandom

/** How the validator's worker hands its result to the validator that started it (menger#137).
  *
  * The worker runs the scene's code, so the scene shares its standard output and can print
  * anything on it, a valid-looking result included. The result therefore travels on a line that
  * starts with a nonce the scene never sees: the parent makes it, writes it to the worker's
  * standard input, and the worker reads it into a local variable before it loads the scene.
  * The parent accepts only the first line that starts with it; every other line is scene output.
  */
private[tools] object WorkerProtocol:

  /** 128 bits, as 32 hexadecimal digits: a scene cannot guess it, and one line cannot spell it
    * by accident. */
  val NonceBytes = 16

  def newNonce(): String =
    val bytes = new Array[Byte](NonceBytes)
    new SecureRandom().nextBytes(bytes)
    bytes.map(b => f"${b & 0xff}%02x").mkString

  /** What every result line starts with: the nonce and a space. */
  def resultPrefix(nonce: String): String = s"$nonce "

  /** The result on one line (`json` has no line break: compact JSON escapes them). It starts with
    * a line break of its own, so that a scene's half-written line already on the stream cannot
    * glue itself to the front of it. */
  def resultLine(nonce: String, json: String): Array[Byte] =
    s"\n${resultPrefix(nonce)}$json\n".getBytes(UTF_8)
