package menger.gdx

import com.badlogic.gdx.graphics.GL20
import menger.objects.Builder
import menger.objects.Composite
import menger.objects.Cube
import menger.objects.Sphere
import menger.objects.Square
import org.scalatest.BeforeAndAfterEach
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Tests for ModelFactory abstraction and geometry model caching logic.
 *
 * Uses MockModelFactory to test caching behavior without requiring LibGDX/LWJGL initialization.
 */
class ModelFactorySuite extends AnyFlatSpec with Matchers with BeforeAndAfterEach:
  given menger.ProfilingConfig = menger.ProfilingConfig.disabled

  override def beforeEach(): Unit =
    // Use mock factory for all tests
    Builder.setModelFactory(ModelFactory.mock)
    // Clear model caches
    clearModelCaches()

  override def afterEach(): Unit =
    // Restore default factory
    Builder.setModelFactory(ModelFactory.default)

  private def clearModelCaches(): Unit =
    // Access the private models map via reflection to clear it
    val sphereClass = Sphere.getClass
    val sphereModelsField = sphereClass.getDeclaredField("models")
    sphereModelsField.setAccessible(true)
    sphereModelsField.get(Sphere) match
      case sphereModels: scala.collection.mutable.Map[_, _] => sphereModels.clear()
      case _ => ()

    val cubeClass = Cube.getClass
    val cubeModelsField = cubeClass.getDeclaredField("models")
    cubeModelsField.setAccessible(true)
    cubeModelsField.get(Cube) match
      case cubeModels: scala.collection.mutable.Map[_, _] => cubeModels.clear()
      case _ => ()

  "MockModelFactory" should "create stub models without LibGDX initialization" in:
    val factory = ModelFactory.mock
    val model = factory.createSphere(1f, 1f, 1f, 32, 16, Builder.WHITE_MATERIAL, Builder.DEFAULT_FLAGS)
    model should not be null
    model.nodes.size should be > 0

  it should "create stub boxes" in:
    val factory = ModelFactory.mock
    val model = factory.createBox(1f, 1f, 1f, GL20.GL_TRIANGLES, Builder.WHITE_MATERIAL, Builder.DEFAULT_FLAGS)
    model should not be null

  it should "create stub rectangles" in:
    val factory = ModelFactory.mock
    val model = factory.createRect(
      -0.5f, -0.5f, 0,
      0.5f, -0.5f, 0,
      0.5f, 0.5f, 0,
      -0.5f, 0.5f, 0,
      0, 0, 1,
      GL20.GL_TRIANGLES,
      Builder.WHITE_MATERIAL,
      Builder.DEFAULT_FLAGS
    )
    model should not be null

  it should "support complex model building" in:
    val factory = ModelFactory.mock
    factory.begin()
    // part() returns null in mock, which is fine for testing caching logic
    factory.end() should not be null

  "Sphere model caching" should "not store models on instantiation" in:
    Sphere.numStoredModels should be(0)
    Sphere()
    Sphere.numStoredModels should be(0)

  it should "store a model when getModel is called" in:
    Sphere.numStoredModels should be(0)
    val sphere = Sphere(com.badlogic.gdx.math.Vector3.Zero, 1)
    sphere.getModel
    Sphere.numStoredModels should be(1)

  it should "reuse cached models for same parameters" in:
    Sphere.numStoredModels should be(0)
    val sphere1 = Sphere(com.badlogic.gdx.math.Vector3.Zero, 1)
    sphere1.getModel
    Sphere.numStoredModels should be(1)

    val sphere2 = Sphere(com.badlogic.gdx.math.Vector3.Zero, 2)
    sphere2.getModel
    // Same divisions, material, primitiveType -> reuse cached model
    Sphere.numStoredModels should be(1)

  it should "create new models for different parameters" in:
    Sphere.numStoredModels should be(0)
    val sphere1 = Sphere(com.badlogic.gdx.math.Vector3.Zero, 1)
    sphere1.getModel
    Sphere.numStoredModels should be(1)

    // Different divisions -> new model
    val sphere2 = Sphere(com.badlogic.gdx.math.Vector3.Zero, 1, divisions = 10)
    sphere2.getModel
    Sphere.numStoredModels should be(2)

  "Cube model caching" should "store one model after first getModel call" in:
    Cube.numStoredModels should be(0)
    val cube = Cube(com.badlogic.gdx.math.Vector3.Zero, 1)
    cube.getModel
    Cube.numStoredModels should be(1)

  it should "store different models for different primitive types" in:
    Cube.numStoredModels should be(0)
    val cube1 = Cube(com.badlogic.gdx.math.Vector3.Zero, 1, primitiveType = GL20.GL_TRIANGLES)
    cube1.getModel
    Cube.numStoredModels should be(1)

    val cube2 = Cube(com.badlogic.gdx.math.Vector3.Zero, 1, primitiveType = GL20.GL_LINES)
    cube2.getModel
    Cube.numStoredModels should be(2)

  "Square" should "create a model" in:
    val square = Square(com.badlogic.gdx.math.Vector3.Zero, 1f)
    square.getModel should have size 1

  it should "support custom material and primitive type" in:
    val square = Square(com.badlogic.gdx.math.Vector3.Zero, 1f, Builder.WHITE_MATERIAL, GL20.GL_LINES)
    square.primitiveType should be(GL20.GL_LINES)
    square.getModel should have size 1

  "Composite" should "return empty model list for empty geometries" in:
    val composite = Composite(geometries = List.empty)
    composite.getModel should be(empty)

  it should "return same model count as single geometry" in:
    val sphere = Sphere(com.badlogic.gdx.math.Vector3.Zero, 1f)
    val composite = Composite(geometries = List(sphere))
    composite.getModel should have size sphere.getModel.size

  it should "combine models from multiple geometries" in:
    val sphere = Sphere(com.badlogic.gdx.math.Vector3.Zero, 1f)
    val cube = Cube(com.badlogic.gdx.math.Vector3.Zero, 1f)
    val composite = Composite(geometries = List(sphere, cube))
    val expectedSize = sphere.getModel.size + cube.getModel.size
    composite.getModel should have size expectedSize

  it should "work with nested composites" in:
    val sphere = Sphere(com.badlogic.gdx.math.Vector3.Zero, 1f)
    val cube = Cube(com.badlogic.gdx.math.Vector3.Zero, 1f)
    val innerComposite = Composite(geometries = List(sphere, cube))
    val square = Square(com.badlogic.gdx.math.Vector3.Zero, 1f)
    val outerComposite = Composite(geometries = List(innerComposite, square))

    val expectedSize = sphere.getModel.size + cube.getModel.size + square.getModel.size
    outerComposite.getModel should have size expectedSize

  "LibGDXModelFactory" should "use actual ModelBuilder" in:
    // This test doesn't run LibGDX, just verifies the factory exists
    val factory = ModelFactory.default
    factory shouldBe a[ModelFactory]
