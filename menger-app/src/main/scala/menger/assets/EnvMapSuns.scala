package menger.assets

import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.ConcurrentHashMap

import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.envmap.EnvMapSun
import menger.dsl.Color
import menger.dsl.EnvMapSunMeasurement
import menger.dsl.EnvMapSunSource
import menger.dsl.Vec3
import menger.failure.Fail

/** The sun of an environment map, measured from the `.hdr` file with `EnvMapSun.measure` (pure
  * JVM: no GPU and no native library, so it also runs in the scene validator) and cached per file.
  * What `Sun.fromEnvMap` builds a `Sun` from, and what the scene converter reads to leave that sun
  * out of the image-based lighting.
  *
  * A measurement reads the whole map into memory (a 4K map is 100 MB of floats) and takes a few
  * seconds, so it is made once per file and kept; the key carries the file's size and time, so an
  * edited map is measured again.
  *
  * Where the map is is decided by `configure`, which installs the lookup into the DSL's
  * `EnvMapSunSource`: relative to `--texture-dir`, under the same [[AssetPaths]] rules as every
  * other asset. Nothing is configured out of the box (tests): no map has a sun and
  * `Sun.fromEnvMap` makes a stand-in sun. The scene validator configures the lenient mode (its
  * sandbox mounts no texture directory): a map that is not there is not an error, `measure`
  * returns `None`, once warned about. A render configures the strict mode, in which a missing map
  * stops the scene with a message saying where it was looked for.
  */
object EnvMapSuns extends LazyLogging:

  private final case class FileKey(file: Path, size: Long, modifiedMillis: Long)

  private val cache = new ConcurrentHashMap[FileKey, EnvMapSunMeasurement]()
  private val warned = ConcurrentHashMap.newKeySet[Path]()

  /** Where maps are looked for, and whether a missing one is an error: `Sun.fromEnvMap` reads
    * its measurements from here from now on.
    */
  def configure(textureDir: String, strict: Boolean): Unit =
    EnvMapSunSource.install(Lookup(textureDir, strict).measure)

  /** Finds maps under `textureDir`; a missing one is an error when `strict`. */
  final case class Lookup(textureDir: String, strict: Boolean):

    /** The sun of the map at `path` (relative to the texture directory), or `None` when the map
      * is not there and the mode is not strict. Fails with a `ConfigurationException` when the
      * path escapes the texture directory or the file is no map with a distinct sun, and with an
      * `EnvironmentException` in strict mode when the file is missing.
      */
    def measure(path: String): Option[EnvMapSunMeasurement] =
      AssetPaths.resolve(textureDir, path) match
        case Left(reason) => Fail.configuration(reason)
        case Right(file) if Files.isRegularFile(file) => Some(measureCached(path, file))
        case Right(file) => missing(path, file)

    private def missing(path: String, file: Path): Option[EnvMapSunMeasurement] =
      if strict then
        Fail.environment(
          s"the environment map '$path' is not there (looked for $file)",
          s"put it under --texture-dir ($textureDir); scripts/fetch-kiara-hdri.sh fetches the Kiara maps")
      else
        if warned.add(file) then
          logger.warn(s"[EnvMapSun] the environment map '$path' is not there (looked for $file): " +
            "its sun is a stand-in")
        None

  private def measureCached(path: String, file: Path): EnvMapSunMeasurement =
    val key = FileKey(file, Files.size(file), Files.getLastModifiedTime(file).toMillis)
    cache.computeIfAbsent(key, _ => measureFile(path, file))

  private def measureFile(path: String, file: Path): EnvMapSunMeasurement =
    val started = System.nanoTime()
    val sun =
      try EnvMapSun.measure(file)
      catch
        case problem @ (_: IllegalArgumentException | _: IOException) =>
          Fail.configuration(s"cannot take the sun from the environment map '$path': ${problem.getMessage}")
    logger.info(
      f"[EnvMapSun] $path: sun toward ${sun.directionToSun}, half-angle ${sun.halfAngleDegrees}%.1f deg, " +
        f"intensity ${sun.intensity}%.2f, ${(System.nanoTime() - started) / 1e6}%.0f ms")
    EnvMapSunMeasurement(
      Vec3(sun.directionToSun(0), sun.directionToSun(1), sun.directionToSun(2)),
      sun.halfAngleDegrees, sun.intensity, Color.fromCommon(sun.color))
