package menger.engines.scene

import java.nio.file.{Files, Path}
import scala.util.{Failure, Success, Try}

/** Optional sidecar metadata for a PBR texture set.
  *
  * Read from `menger-textureset.json` in the texture set directory.
  * All fields optional — missing fields use material preset defaults.
  */
case class TextureSetMetadata(
  ior: Option[Float] = None,
  uvScale: Option[Float] = None
)

object TextureSetMetadata:

  /** Load metadata from the set directory, if the sidecar exists.
    * Returns empty metadata (all None) if the file doesn't exist.
    */
  def load(dir: Path): Try[TextureSetMetadata] =
    val sidecar = dir.resolve("menger-textureset.json")
    if !Files.exists(sidecar) then
      Success(TextureSetMetadata())
    else
      Try:
        val content = Files.readString(sidecar)
        val result = parseSimple(content)
        validate(result)
        result

  /** Minimal JSON parser for the sidecar format.
    * Handles: {"ior": 1.45, "uvScale": 2.0}
    * No nesting, no arrays — just top-level float fields.
    */
  private def parseSimple(json: String): TextureSetMetadata =
    val floatPattern = """"(ior|uvScale)"\s*:\s*([0-9]+\.?[0-9]*)""".r
    val matches = floatPattern.findAllMatchIn(json).toList
    val ior = matches.collectFirst:
      case m if m.group(1) == "ior" => m.group(2).toFloat
    val uvScale = matches.collectFirst:
      case m if m.group(1) == "uvScale" => m.group(2).toFloat
    TextureSetMetadata(ior, uvScale)

  private def validate(meta: TextureSetMetadata): Unit =
    meta.ior.foreach: v =>
      require(v >= 1.0f, s"IOR must be >= 1.0, got $v")
    meta.uvScale.foreach: v =>
      require(v > 0f, s"uvScale must be > 0, got $v")
