package menger.engines.scene

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging
import menger.ObjectSpec
import menger.optix.OptiXRenderer

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
 *   builder.validate(specs, maxInstances) match
 *     case Left(error) => // Handle validation error
 *     case Right(_) => builder.buildScene(specs, renderer)
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
