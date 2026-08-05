package io.github.lene.optix

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files

import scala.collection.concurrent.TrieMap

import com.typesafe.scalalogging.LazyLogging
import menger.common.Material
import menger.common.Vector
import menger.common.x
import menger.common.y
import menger.common.z

/** Menger-specific OptiX renderer.
 *
 *  Extends [[OptiXRenderer]] and supplies the 4D fractal primitives (menger4d,
 *  sierpinski4d, hexadecachoron4d) through optix-jni's *generic* custom-geometry
 *  SPI — optix-jni no longer knows these types (Sprint 35, AD-24). Each type's
 *  shaders live in `menger_4d.ptx` (built by menger-geometry) and are registered
 *  once, lazily, after the renderer is initialized; instances then travel the SPI's
 *  `addCustomGeometryInstance` + `setInstanceMaterial` path, with per-frame
 *  projection changes applied in place via `updateCustomGeometryInstanceData`.
 *
 *  The recursive 3D IAS sponge stayed generic in optix-jni and is inherited, not
 *  overridden. libmengergeometry.so is still loaded, now for video decoding only
 *  (Task 1.3 moved caustics wholly into optix-jni).
 *
 *  Use [[MengerRenderer.apply]] to construct. Both native libraries load via the
 *  shared [[NativeLibrary]] loader; load order is irrelevant since Task 1.3, which
 *  left libmengergeometry.so no longer linking any optix-jni symbols.
 */
class MengerRenderer extends OptiXRenderer with LazyLogging:

  import MengerRenderer.*

  // 4D per-instance blob is a 96-byte struct (see MengerGeometryData.h). Every field
  // is 4-byte aligned; the layout below MUST match the C++ structs exactly.
  private val BlobBytes = 96

  // Registered geometry-type ids, allocated on first use (registration needs an
  // initialized renderer — the module is built in the renderer's OptiX context).
  private lazy val ptxBytes: Array[Byte] = loadMenger4dPtx()
  private lazy val menger4dType: Int =
    registerCustomGeometry(ptxBytes, "__intersection__menger4d", "__closesthit__menger4d",
      "__closesthit__menger4d_shadow", "__anyhit__menger4d_shadow")
  private lazy val sierpinski4dType: Int =
    registerCustomGeometry(ptxBytes, "__intersection__sierpinski4d", "__closesthit__sierpinski4d",
      "__closesthit__sierpinski4d_shadow", "__anyhit__sierpinski4d_shadow")
  private lazy val hexadecachoron4dType: Int =
    registerCustomGeometry(ptxBytes, "__intersection__hexadecachoron4d", "__closesthit__hexadecachoron4d",
      "__closesthit__hexadecachoron4d_shadow", "__anyhit__hexadecachoron4d_shadow")

  // Immutable per-instance fields, kept so a projection update can rebuild the blob
  // (only eye/screen/rotation change per frame). lastWord is the struct's trailing
  // 4 bytes: dist_threshold (int) for menger4d, hit_bias (float bits) for the others.
  private val instanceState = TrieMap[Int, InstanceState]()

  /** Adds a GPU-projected 4D Menger sponge instance. Rotation angles are degrees. */
  def addMenger4DInstance(
    level: Int, distanceThreshold: Int, position: Vector[3], scale: Float,
    eyeW: Float, screenW: Float, rotXW: Float, rotYW: Float, rotZW: Float,
    material: Material
  ): Int =
    add4D(menger4dType, position, scale, level, distanceThreshold,
      eyeW, screenW, rotXW, rotYW, rotZW, material)

  /** Adds a GPU-projected 4D Sierpinski instance. Rotation angles are degrees. */
  def addSierpinski4DInstance(
    level: Int, position: Vector[3], scale: Float,
    eyeW: Float, screenW: Float, rotXW: Float, rotYW: Float, rotZW: Float,
    material: Material
  ): Int =
    val hitBias = if material.color.a < OpaqueAlpha then SierpinskiHitBias else 0.0f
    add4D(sierpinski4dType, position, scale, level, floatBits(hitBias),
      eyeW, screenW, rotXW, rotYW, rotZW, material)

  /** Adds a GPU-projected 4D hexadecachoron instance. Rotation angles are degrees. */
  def addHexadecachoron4DInstance(
    level: Int, position: Vector[3], scale: Float,
    eyeW: Float, screenW: Float, rotXW: Float, rotYW: Float, rotZW: Float,
    material: Material
  ): Int =
    val hitBias = if material.color.a < OpaqueAlpha then HexadecachoronHitBias else 0.0f
    add4D(hexadecachoron4dType, position, scale, level, floatBits(hitBias),
      eyeW, screenW, rotXW, rotYW, rotZW, material)

  /** Updates projection parameters for any GPU-projected 4D IFS instance (menger4d /
    * sierpinski4d / hexadecachoron4d). The per-type distinction lives only in the
    * immutable add-time fields (level, threshold/hit-bias), which the update rebuilds
    * from the registered instance state — so one update path serves all three (F9). */
  def update4DProjection(
    instanceId: Int, eyeW: Float, screenW: Float, rotXW: Float, rotYW: Float, rotZW: Float
  ): Unit =
    update4D(instanceId, eyeW, screenW, rotXW, rotYW, rotZW)

  private def add4D(
    typeId: Int, position: Vector[3], scale: Float, level: Int, lastWord: Int,
    eyeW: Float, screenW: Float, rotXW: Float, rotYW: Float, rotZW: Float,
    material: Material
  ): Int =
    val rotation = composeRotationXwYwZw(rotXW, rotYW, rotZW)
    val blob = packBlob(position, scale, rotation, eyeW, screenW, level, lastWord)
    // Conservative AABB: the projected fractal stays within `scale` of its centre.
    val aabbMin = Array(position.x - scale, position.y - scale, position.z - scale)
    val aabbMax = Array(position.x + scale, position.y + scale, position.z + scale)
    val instanceId = addCustomGeometryInstance(typeId, aabbMin, aabbMax, IdentityTransform, blob)
    if instanceId < 0 then instanceId
    else
      val (cauchyA, cauchyB) = Material.cauchyCoefficients(material.ior, material.dispersion)
      setInstanceMaterial(
        instanceId,
        material.color.r, material.color.g, material.color.b, material.color.a,
        material.ior, material.roughness, material.metallic, material.specular,
        material.emission, material.filmThickness, cauchyA, cauchyB)
      instanceState(instanceId) = InstanceState(position, scale, level, lastWord)
      instanceId

  private def update4D(
    instanceId: Int, eyeW: Float, screenW: Float, rotXW: Float, rotYW: Float, rotZW: Float
  ): Unit =
    require(instanceId >= 0, s"instanceId must be non-negative, got $instanceId")
    val state = instanceState.getOrElse(instanceId,
      throw IllegalArgumentException(s"No 4D instance registered with id $instanceId"))
    val rotation = composeRotationXwYwZw(rotXW, rotYW, rotZW)
    val blob = packBlob(state.position, state.scale, rotation, eyeW, screenW, state.level, state.lastWord)
    val rc = updateCustomGeometryInstanceData(instanceId, blob)
    require(rc == 0, s"updateCustomGeometryInstanceData failed with code $rc (instanceId=$instanceId)")

  private def packBlob(
    position: Vector[3], scale: Float, rotation: Array[Float],
    eyeW: Float, screenW: Float, level: Int, lastWord: Int
  ): Array[Byte] =
    val buf = ByteBuffer.allocate(BlobBytes).order(ByteOrder.LITTLE_ENDIAN)
    buf.putFloat(position.x).putFloat(position.y).putFloat(position.z)
    buf.putFloat(scale)
    rotation.foreach(buf.putFloat)
    buf.putFloat(eyeW).putFloat(screenW)
    buf.putInt(level)
    buf.putInt(lastWord)
    buf.array

  override def ensureAvailable(): OptiXRenderer =
    if !MengerRenderer.isLibraryLoaded then
      throw OptiXNotAvailableException(
        "Menger native library (mengergeometry) failed to load"
      )
    super.ensureAvailable()

object MengerRenderer:
  private val libraryName = "mengergeometry"

  // Alpha at/above which a fractal is opaque (matches the old native hit_bias gate).
  private val OpaqueAlpha = 0.999f
  private val SierpinskiHitBias = 9e-4f
  private val HexadecachoronHitBias = 0.01f

  private val IdentityTransform: Array[Float] =
    Array(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f)

  private def floatBits(f: Float): Int = java.lang.Float.floatToRawIntBits(f)

  private val libraryLoaded: Boolean = NativeLibrary.load(libraryName)

  def isLibraryLoaded: Boolean = libraryLoaded

  def apply(): MengerRenderer = new MengerRenderer()

  /** Views an [[OptiXRenderer]] as a [[MengerRenderer]] for 4D calls. The app types
    * the renderer as OptiXRenderer everywhere but always builds a MengerRenderer;
    * fully retyping that call path is Ph4 (F8/F13). */
  def of(renderer: OptiXRenderer): MengerRenderer = renderer match
    case m: MengerRenderer => m
    case _ => throw OptiXNotAvailableException(
      "4D geometry requires a MengerRenderer (got a plain OptiXRenderer)")

  /** Compose the 4D rotation matrix as R_xw * (R_yw * R_zw), row-major float[16].
    * Angles are DEGREES (ports the old native compose_rotation_xw_yw_zw exactly;
    * matches Rotation.scala). */
  private[optix] def composeRotationXwYwZw(degXw: Float, degYw: Float, degZw: Float): Array[Float] =
    val rxw = rotationPlane(0, 3, degXw)
    val ryw = rotationPlane(1, 3, degYw)
    val rzw = rotationPlane(2, 3, degZw)
    mat4Mul(rxw, mat4Mul(ryw, rzw))

  private def rotationPlane(row: Int, col: Int, deg: Float): Array[Float] =
    val rad = deg * 0.017453292519943295f
    val c = math.cos(rad).toFloat
    val s = math.sin(rad).toFloat
    val out = Array.fill(16)(0.0f)
    for i <- 0 until 4 do out(i * 4 + i) = 1.0f
    out(row * 4 + row) = c
    out(row * 4 + col) = s
    out(col * 4 + row) = -s
    out(col * 4 + col) = c
    out

  private def mat4Mul(a: Array[Float], b: Array[Float]): Array[Float] =
    val r = Array.fill(16)(0.0f)
    for i <- 0 until 4; j <- 0 until 4 do
      var sum = 0.0f
      for k <- 0 until 4 do sum += a(i * 4 + k) * b(k * 4 + j)
      r(i * 4 + j) = sum
    r

  private final case class InstanceState(position: Vector[3], scale: Float, level: Int, lastWord: Int)

  /** Load the registrable 4D shader module (menger_4d.ptx), from the classpath first
    * and the sbt native build output as a fallback (sbt run / test). */
  private def loadMenger4dPtx(): Array[Byte] =
    val platform = NativeLibrary.platform()
    val resourcePath = s"/native/$platform/menger_4d.ptx"
    Option(getClass.getResourceAsStream(resourcePath)) match
      case Some(stream) => try stream.readAllBytes() finally stream.close()
      case None =>
        val candidates = List(
          s"menger-geometry/target/native/$platform/bin/menger_4d.ptx",
          s"target/native/$platform/bin/menger_4d.ptx")
        candidates.map(java.nio.file.Paths.get(_)).find(Files.exists(_)) match
          case Some(path) => Files.readAllBytes(path)
          case None => throw IllegalStateException(
            s"menger_4d.ptx not found on classpath ($resourcePath) or in $candidates")
