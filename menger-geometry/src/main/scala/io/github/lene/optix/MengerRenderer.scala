package io.github.lene.optix

import java.io.FileOutputStream
import java.io.InputStream
import java.nio.file.Files

import scala.util.Failure
import scala.util.Success
import scala.util.Try
import scala.util.control.Exception.catching

import com.typesafe.scalalogging.LazyLogging

/** Menger-specific OptiX renderer.
 *
 *  Extends [[OptiXRenderer]] and routes the 4D geometry @native methods through
 *  libmengergeometry.so (MengerJNIBindings.cpp) instead of liboptixjni.so.
 *  All other rendering functionality is inherited from [[OptiXRenderer]].
 *
 *  Use [[MengerRenderer.apply]] to construct; it ensures both native libraries
 *  are loaded in the correct order (optixjni before mengergeometry).
 */
class MengerRenderer extends OptiXRenderer:

  // Override 4D geometry @native methods — JVM dispatches to
  // Java_io_github_lene_optix_MengerRenderer_* in libmengergeometry.so

  @native override private[optix] def addRecursiveIASSpongeInstanceNative(
    level: Int,
    transform: Array[Float],
    r: Float, g: Float, b: Float, a: Float,
    ior: Float, roughness: Float, metallic: Float, specular: Float, emission: Float,
    textureIndex: Int,
    filmThickness: Float
  ): Int

  @native override private[optix] def addMenger4DInstanceNative(
    level: Int,
    distanceThreshold: Int,
    x: Float, y: Float, z: Float, scale: Float,
    eyeW: Float, screenW: Float,
    rotXW: Float, rotYW: Float, rotZW: Float,
    r: Float, g: Float, b: Float, a: Float,
    ior: Float, roughness: Float, metallic: Float, specular: Float, emission: Float,
    filmThickness: Float
  ): Int

  @native override private[optix] def updateMenger4DProjectionNative(
    instanceId: Int,
    eyeW: Float, screenW: Float,
    rotXW: Float, rotYW: Float, rotZW: Float
  ): Int

  @native override private[optix] def addSierpinski4DInstanceNative(
    level: Int,
    x: Float, y: Float, z: Float,
    scale: Float, eyeW: Float, screenW: Float,
    rotXW: Float, rotYW: Float, rotZW: Float,
    r: Float, g: Float, b: Float, a: Float,
    ior: Float, roughness: Float, metallic: Float,
    specular: Float, emission: Float, filmThickness: Float
  ): Int

  @native override private[optix] def updateSierpinski4DProjectionNative(
    instanceId: Int,
    eyeW: Float, screenW: Float,
    rotXW: Float, rotYW: Float, rotZW: Float
  ): Int

  @native override private[optix] def addHexadecachoron4DInstanceNative(
    level: Int,
    x: Float, y: Float, z: Float,
    scale: Float, eyeW: Float, screenW: Float,
    rotXW: Float, rotYW: Float, rotZW: Float,
    r: Float, g: Float, b: Float, a: Float,
    ior: Float, roughness: Float, metallic: Float,
    specular: Float, emission: Float, filmThickness: Float
  ): Int

  @native override private[optix] def updateHexadecachoron4DProjectionNative(
    instanceId: Int,
    eyeW: Float, screenW: Float,
    rotXW: Float, rotYW: Float, rotZW: Float
  ): Int

  override def ensureAvailable(): OptiXRenderer =
    if !MengerRenderer.isLibraryLoaded then
      throw OptiXNotAvailableException(
        "Menger native library (mengergeometry) failed to load"
      )
    super.ensureAvailable()

object MengerRenderer extends LazyLogging:
  private val libraryName = "mengergeometry"

  // Ensure optixjni is loaded before mengergeometry (mengergeometry has
  // undefined symbols from liboptixjni.so resolved at runtime).
  private val _optixJniInit: Boolean = OptiXRenderer.isLibraryLoaded

  private val libraryLoaded: Boolean = loadNativeLibrary().isSuccess

  def isLibraryLoaded: Boolean = libraryLoaded

  def apply(): MengerRenderer = new MengerRenderer()

  private def detectPlatform(): Try[String] =
    val os   = System.getProperty("os.name").toLowerCase
    val arch = System.getProperty("os.arch").toLowerCase
    (os, arch) match
      case (o, a) if o.contains("linux") && (a.contains("amd64") || a.contains("x86_64")) =>
        Success("x86_64-linux")
      case _ =>
        Failure(new UnsupportedOperationException(s"Unsupported platform: $os/$arch"))

  private def copyStream(stream: InputStream, out: FileOutputStream): Try[Unit] = Try:
    val buffer = new Array[Byte](8192)
    @scala.annotation.tailrec
    def loop(): Unit =
      stream.read(buffer) match
        case -1 => ()
        case n  => out.write(buffer, 0, n); loop()
    loop()

  private def extractAndLoad(stream: InputStream): Try[Unit] = Try:
    val tempFile = Files.createTempFile(s"lib$libraryName", ".so")
    tempFile.toFile.deleteOnExit()
    val out = new FileOutputStream(tempFile.toFile)
    try copyStream(stream, out).get
    finally { out.close(); stream.close() }
    System.load(tempFile.toAbsolutePath.toString)

  private def extractPTX(platform: String): Try[Unit] = Try:
    val resourcePath = s"/native/$platform/optix_shaders_menger.ptx"
    Option(getClass.getResourceAsStream(resourcePath)) match
      case Some(ptxStream) =>
        val ptxDir  = new java.io.File("target/native/x86_64-linux/bin")
        ptxDir.mkdirs()
        val ptxFile = new java.io.File(ptxDir, "optix_shaders_menger.ptx")
        val ptxOut  = new FileOutputStream(ptxFile)
        try copyStream(ptxStream, ptxOut).get
        finally { ptxOut.close(); ptxStream.close() }
      case None =>
        logger.debug(s"Menger PTX resource not found: $resourcePath")

  private def loadFromClasspath(platform: String): Try[Unit] =
    val resourcePath = s"/native/$platform/lib$libraryName.so"
    for
      stream <- Option(getClass.getResourceAsStream(resourcePath))
        .toRight(new IllegalStateException(s"Library resource not found: $resourcePath"))
        .toTry
      _ <- extractAndLoad(stream)
      _ <- extractPTX(platform)
    yield ()

  private def loadNativeLibrary(): Try[Unit] =
    catching(classOf[UnsatisfiedLinkError])
      .withTry(System.loadLibrary(libraryName))
      .recoverWith { case _: UnsatisfiedLinkError =>
        detectPlatform().flatMap(loadFromClasspath)
      }
      .recoverWith { case e: Exception =>
        logger.error(s"Failed to load native library '$libraryName'", e)
        Failure(e)
      }
