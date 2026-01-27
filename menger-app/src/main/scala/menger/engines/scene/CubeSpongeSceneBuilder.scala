package menger.engines.scene

import scala.util.Try

import com.badlogic.gdx.math.Vector3
import menger.ObjectSpec
import menger.common.Const
import menger.common.TransformUtil
import menger.objects.Cube
import menger.objects.CubeSpongeGenerator
import menger.optix.Material
import menger.optix.OptiXRenderer

/**
 * Scene builder for cube-sponge fractals.
 *
 * Creates scenes with cube-sponge fractals, where each spec generates multiple cube instances:
 * - Each cube-sponge spec generates 20^level cube instances
 * - All instances share the same base cube mesh (scale 1.0)
 * - Instance transforms position and scale each cube appropriately
 *
 * Key characteristics:
 * - Base cube mesh set once (shared by all instances)
 * - Instance count = sum of 20^level for each spec (exponential growth)
 * - Instance limit validation critical (level 3 = 8000 cubes!)
 * - Success logging suppressed (too verbose for thousands of cubes)
 * - Each spec can have different level/position/material
 *
 * Ported from OptiXEngine.setupCubeSponges() (lines 492-500),
 * setupBaseCubeMesh() (lines 436-439),
 * addAllCubeInstances() (lines 441-447),
 * addCubeInstancesForSpec() (lines 449-472),
 * addSingleCubeInstance() (lines 474-490).
 */
class CubeSpongeSceneBuilder extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then
      Left("Object specs list cannot be empty")
    else if !specs.forall(_.objectType == "cube-sponge") then
      Left("All objects must be cube-sponges for CubeSpongeSceneBuilder")
    else if !specs.forall(_.level.isDefined) then
      Left("All cube-sponge specs must have level parameter defined")
    else
      val totalInstances = calculateInstanceCount(specs)
      if totalInstances > maxInstances then
        Left(s"cube-sponge specs generate $totalInstances total instances, " +
          s"exceeding max instances limit of $maxInstances. " +
          "Reduce sponge levels or use --max-instances to increase the limit.")
      else
        Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] =
    val totalInstances = calculateInstanceCount(specs)
    logger.debug(s"Setting up ${specs.length} cube-sponge(s) generating $totalInstances total cube instances")

    for
      _ <- setupBaseCubeMesh(renderer)
    yield
      specs.foreach(addCubeInstancesForSpec(_, renderer))

  private def setupBaseCubeMesh(renderer: OptiXRenderer): Try[Unit] = Try:
    // Create base cube mesh centered at origin with scale 1.0 (shared by all instances)
    val baseCube = Cube(center = Vector3(0f, 0f, 0f), scale = 1.0f)
    renderer.setTriangleMesh(baseCube.toTriangleMesh)

  private def addCubeInstancesForSpec(spec: ObjectSpec, renderer: OptiXRenderer): Unit =
    require(spec.level.isDefined, "cube-sponge requires level")
    val level = spec.level.get.toInt
    val material = MaterialExtractor.extract(spec)

    // Generate all cube transforms using CubeSpongeGenerator
    val generator = CubeSpongeGenerator(
      center = Vector3(spec.x, spec.y, spec.z),
      size = spec.size,
      level = level
    )

    logger.debug(s"Generating ${generator.cubeCount} cube instances for level $level cube-sponge at (${spec.x}, ${spec.y}, ${spec.z})")

    // Add each cube as an instance
    generator.generateTransforms.foreach { case (position, scale) =>
      addSingleCubeInstance(position, scale, material, renderer)
    }

    logger.debug(s"Added ${generator.cubeCount} cube instances for cube-sponge")

  private def addSingleCubeInstance(
    position: Vector3,
    scale: Float,
    material: Material,
    renderer: OptiXRenderer
  ): Unit =
    val transform = TransformUtil.createScaleTranslation(
      scale, position.x, position.y, position.z
    )

    val instanceId = renderer.addTriangleMeshInstance(transform, material)

    instanceId match
      case None =>
        logger.error(s"Failed to add cube instance at position=($position), scale=$scale")
      case Some(_) =>
        // Success - don't log each instance (too verbose for 8000+ cubes)

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    // All cube-sponges are compatible with each other
    spec1.objectType == "cube-sponge" && spec2.objectType == "cube-sponge"

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    specs.map { spec =>
      require(spec.level.isDefined, "cube-sponge requires level")
      Math.pow(Const.Engine.cubesPerSpongeLevel, spec.level.get.toInt).toLong
    }.sum
