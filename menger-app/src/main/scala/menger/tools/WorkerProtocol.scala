package menger.tools

import java.nio.charset.StandardCharsets.UTF_8
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

/** How the validator's worker hands its result to the validator that started it (menger#137).
  *
  * The worker runs the scene's code, so the scene shares its standard output and can print
  * anything on it, a valid-looking result included. The result therefore travels on a line that
  * starts with a nonce the scene never sees: the parent makes it, writes it to the worker's
  * standard input, and the worker reads it into a local variable before it loads the scene.
  * The parent accepts only the first line that starts with it; every other line is scene output.
  *
  * The result is also signed: the line carries an HMAC-SHA256 of the result, keyed by the nonce.
  * A pipe write is atomic only up to `PIPE_BUF` (4096 bytes) and the scene's threads share the
  * stream, so a scene thread can land its own bytes inside a long result line; the parent
  * rejects such a line (a verdict of `refused`) instead of reading what the scene wrote into it.
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

  private val MacAlgorithm = "HmacSHA256"

  /** The HMAC of `json`, keyed by the nonce, as lower-case hexadecimal. */
  def mac(nonce: String, json: String): String =
    val hmac = Mac.getInstance(MacAlgorithm)
    hmac.init(new SecretKeySpec(nonce.getBytes(UTF_8), MacAlgorithm))
    hmac.doFinal(json.getBytes(UTF_8)).map(b => f"${b & 0xff}%02x").mkString

  /** The result on one line: the nonce, the HMAC of `json`, then `json` (which has no line break:
    * compact JSON escapes them). It starts with a line break of its own, so that a scene's
    * half-written line already on the stream cannot glue itself to the front of it. */
  def resultLine(nonce: String, json: String): Array[Byte] =
    s"\n${resultPrefix(nonce)}${mac(nonce, json)} $json\n".getBytes(UTF_8)

  /** The result in what follows the nonce on a result line (`<hmac> <json>`), or None when the
    * HMAC is missing or does not match: something other than the worker wrote into the line. */
  def authenticated(nonce: String, payload: String): Option[String] =
    payload.split(" ", 2) match
      case Array(received, json) if MessageDigest.isEqual(
            received.getBytes(UTF_8), mac(nonce, json).getBytes(UTF_8)) => Some(json)
      case _ => None
