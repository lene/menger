package menger

import java.nio.file.Path
import java.nio.file.Paths

/** Resolves a user-supplied asset path (texture, texture-set, video, env map) strictly inside
  * `--texture-dir`. An absolute path or a `..` escape is a security/config error, not a file
  * lookup to attempt — path-traversal exposure flagged in the Sprint 37 usability review (T1#1).
  */
object AssetPaths:

  final case class AssetPathException(message: String) extends RuntimeException(message)

  /** @return `Right(path)` inside `baseDir` (symlink-resolved if the file exists), or
    * `Left(reason)` naming the offending path and `--texture-dir`.
    */
  def resolve(baseDir: String, path: String): Either[String, Path] =
    val requested = Paths.get(path)
    if requested.isAbsolute then
      Left(s"asset path must be relative to --texture-dir, got an absolute path: $path")
    else
      val base = Paths.get(baseDir).toAbsolutePath.normalize
      val candidate = base.resolve(requested).normalize
      if !candidate.startsWith(base) then
        Left(s"asset path '$path' escapes --texture-dir ($baseDir)")
      else if candidate.toFile.exists then
        val realBase = base.toRealPath()
        val realCandidate = candidate.toRealPath()
        if !realCandidate.startsWith(realBase) then
          Left(s"asset path '$path' resolves outside --texture-dir ($baseDir) via a symlink")
        else
          Right(realCandidate)
      else
        Right(candidate)

  /** Same as [[resolve]], throwing [[AssetPathException]] on the [[Left]] case — for call
    * sites that already wrap their body in `Try`/`try` and log the failure there.
    */
  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  def resolveOrThrow(baseDir: String, path: String): Path =
    resolve(baseDir, path).fold(msg => throw AssetPathException(msg), identity)
