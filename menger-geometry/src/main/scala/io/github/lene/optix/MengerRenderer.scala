package io.github.lene.optix

import scala.util.Try

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
  // Ensure optixjni is loaded before mengergeometry (mengergeometry has
  // undefined symbols from liboptixjni.so resolved at runtime).
  private val _optixJniInit: Boolean = OptiXRenderer.isLibraryLoaded

  private val libraryLoaded: Boolean =
    Try(System.loadLibrary("mengergeometry"))
      .recover { case e: UnsatisfiedLinkError =>
        logger.error(s"Failed to load native library mengergeometry: ${e.getMessage}")
      }
      .isSuccess

  def isLibraryLoaded: Boolean = libraryLoaded

  def apply(): MengerRenderer = new MengerRenderer()
