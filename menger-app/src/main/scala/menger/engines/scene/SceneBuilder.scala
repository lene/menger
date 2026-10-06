package menger.engines.scene

import scala.util.Failure
import scala.util.Try

import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.common.TransformUtil
import menger.common.ValidationException

/**
 * Strategy trait for building different scene types in OptiX.
 *
 * Each strategy encapsulates:
 * - Validation logic for object specs
 * - Geometry setup
 * - Instance creation
 * - Compatibility checking
 *
 * Implementations:
 * - SphereSceneBuilder: Multiple sphere instances
 * - TriangleMeshSceneBuilder: Multiple triangle mesh instances with optional textures
 * - CubeSpongeSceneBuilder: Multiple cube-sponge fractals (each generates many instances)
 *
 * Usage:
 * {{{
 *   val builder = SphereSceneBuilder()
 *   builder.validateAndBuild(specs, renderer, maxInstances)
 * }}}
 */
trait SceneBuilder extends LazyLogging:

  /**
   * Validates that the given object specs are compatible with this scene builder.
   *
   * Checks may include:
   * - Spec list is non-empty
   * - All specs are compatible with this builder's scene type
   * - Total instance count doesn't exceed maxInstances limit
   * - Required parameters are present (e.g., level for sponges)
   * - Specs are mutually compatible (e.g., same geometry type for meshes)
   *
   * @param specs List of object specifications
   * @param maxInstances Maximum number of instances allowed
   * @return Left(error) if validation fails, Right(()) if valid
   */
  def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit]

  /**
   * Builds the scene by configuring geometry and adding instances.
   *
   * Must be called after validate() succeeds. Typical workflow:
   * 1. Set base geometry (if applicable)
   * 2. Load resources (e.g., textures)
   * 3. Add all instances with transforms and materials
   *
   * @param specs List of object specifications (pre-validated)
   * @param renderer OptiX renderer to configure
   * @param maxInstances Maximum number of instances (may be auto-adjusted)
   * @return Try[Unit] - Success if scene built successfully, Failure otherwise
   */
  def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit]

  final def validateAndBuild(
    specs: List[ObjectSpec],
    renderer: OptiXRenderer,
    maxInstances: Int
  ): Try[Unit] =
    validate(specs, maxInstances) match
      case Left(error) =>
        Failure(ValidationException(error, "objectSpecs", specs.map(_.objectType)))
      case Right(_) =>
        buildScene(specs, renderer, maxInstances)

  /**
   * Checks if two object specs are compatible for this scene type.
   *
   * Compatibility rules vary by scene type:
   * - Spheres: All spheres are compatible
   * - Triangle meshes: Same geometry type + matching parameters (level, 4D params)
   * - Cube sponges: All cube-sponges are compatible
   *
   * @param spec1 First object spec
   * @param spec2 Second object spec
   * @return true if specs can coexist in the same scene, false otherwise
   */
  def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean

  /**
   * Calculates the total number of instances that will be created.
   *
   * For most types this is specs.length (1:1 mapping).
   * For cube-sponge, each spec generates many instances (20^level).
   *
   * @param specs List of object specifications
   * @return Total instance count
   */
  def calculateInstanceCount(specs: List[ObjectSpec]): Long

  /**
   * Calculate exact number of instances required for the given specs.
   *
   * Used for auto-adjustment of maxInstances before validation.
   * Default implementation returns 0 (no auto-adjustment needed).
   * Override in builders that need dynamic instance calculation.
   *
   * @param specs List of object specifications
   * @return Exact number of instances required
   */
  def calculateRequiredInstances(specs: List[ObjectSpec]): Int = 0

  /** Apply procedural texture, PBR map textures, and image texture to an already-created instance.
    * Centralises the wiring that was previously duplicated in every builder.
    * The imageTexture field is used by cone/plane via image_texture_index (Task 21.6). */
  protected final def applyInstanceTextures(
    id: InstanceId,
    spec: ObjectSpec,
    textureIndices: Map[String, Int],
    renderer: OptiXRenderer
  ): Unit =
    val rawId = InstanceId.raw(id)
    SceneBuilder.objectFrame(spec).foreach(renderer.setObjectFrame(rawId, _))
    if spec.proceduralType != 0 then
      renderer.setProceduralTexture(rawId, spec.proceduralType, spec.proceduralScale)
    // Resolve map indices from spec fields, falling back to texture set if set is defined
    val setPrefix = spec.textureSet.map(s => s"set:$s:")
    val normalIdx    = resolveMapIndex(spec.normalMap, "normal", setPrefix, textureIndices)
    val roughnessIdx = resolveMapIndex(spec.roughnessMap, "roughness", setPrefix, textureIndices)
    val metallicIdx  = resolveMapIndex(spec.metallicMap, "metallic", setPrefix, textureIndices)
    val aoIdx        = resolveMapIndex(spec.aoMap, "ao", setPrefix, textureIndices)
    val heightIdx    = resolveMapIndex(spec.heightMap, "height", setPrefix, textureIndices)
    if normalIdx >= 0 || roughnessIdx >= 0 || metallicIdx >= 0 || aoIdx >= 0 || heightIdx >= 0 then
      renderer.setMapTextures(rawId, normalIdx, roughnessIdx, metallicIdx, aoIdx, heightIdx)
    val imageIdx = spec.imageTextureKey.flatMap(textureIndices.get).getOrElse(-1)
    if imageIdx >= 0 then
      renderer.setImageTexture(rawId, imageIdx)

  private def resolveMapIndex(
    explicitFile: Option[String],
    mapType: String,
    setPrefix: Option[String],
    indices: Map[String, Int]
  ): Int =
    explicitFile.flatMap(indices.get)  // Explicit override wins
      .orElse(setPrefix.flatMap(p => indices.get(p + mapType)))  // Texture set fallback
      .getOrElse(-1)

  protected final def requireInstanceId(rawId: Int, operation: => String): InstanceId =
    InstanceId.fromNative(rawId, operation)

object SceneBuilder:
  /** The object's frame for optix-jni's `setObjectFrame`: the row-major 3x4 world -> local
    * transform mapping the object's own box onto [0,1]^3, undoing the instance transform the
    * builders use (`TransformUtil.createEulerRotationScaleTranslation`: R = Rz Ry Rx, then
    * `pos`). It drives xyz_rgb_local (F63) and a transparent object's shadow (F67). A sphere's
    * `size` is its radius, every other object spans `pos` +- size / 2; a 4D object's box is
    * taken before projection, so it is approximate there. `None` for a plane (unbounded). */
  def objectFrame(spec: ObjectSpec): Option[Array[Float]] =
    Option.when(spec.objectType != "plane" && spec.size > 0f) {
      val halfExtent = if spec.objectType == "sphere" then spec.size else spec.size / 2f
      val k = 1f / (2f * halfExtent)
      val r = TransformUtil.createEulerRotationScaleTranslation(
        spec.rotX, spec.rotY, spec.rotZ, 1f, 0f, 0f, 0f
      )
      // Row i of R^T is column i of R: r(i), r(4 + i), r(8 + i).
      (0 until 3).flatMap { i =>
        val (a, b, c) = (r(i), r(4 + i), r(8 + i))
        Seq(k * a, k * b, k * c, 0.5f - k * (a * spec.x + b * spec.y + c * spec.z))
      }.toArray
    }
