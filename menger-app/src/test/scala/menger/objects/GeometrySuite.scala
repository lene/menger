package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.math.Vector3
import org.scalatest.Tag
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers


object GdxTest extends Tag("GdxTest"):  // needs Gdx to be available
  given menger.ProfilingConfig = menger.ProfilingConfig.disabled

class GeometrySuite extends AnyFlatSpec with Matchers:
  given menger.ProfilingConfig = menger.ProfilingConfig.disabled
  private val loadingLWJGLSucceeds: Boolean = LWJGLLoadChecker.loadingLWJGLSucceeds
  private val ORIGIN = Vector3(0, 0, 0)

  "instantiating a sphere" should "not store a model" taggedAs GdxTest in:
    Sphere.numStoredModels should be (0)
    Sphere()
    Sphere.numStoredModels should be(0)

  "calling at() on a sphere" should "store a model" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    Sphere.numStoredModels should be(0)
    Sphere(ORIGIN, 1).getModel
    Sphere.numStoredModels should be(1)

  "calling at() on different spheres" should "store only one model" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    Sphere.numStoredModels should be(1)
    Sphere(ORIGIN, 1).getModel
    Sphere(ORIGIN, 2).getModel
    Sphere.numStoredModels should be(1)

  "calling at() on sphere with different parameters" should "store two models" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    Sphere.numStoredModels should be(1)
    Sphere(ORIGIN, 1, divisions = 10).getModel
    Sphere.numStoredModels should be(2)

  "sphere toString" should "return class name" in:
    Sphere().toString should be("Sphere")

  "square" should "be one model" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    Square(ORIGIN, 1).getModel should have size 1

  it should "instantiate with material and primitiveType" taggedAs GdxTest in:
    Square(Vector3.Zero, 1f, Builder.WHITE_MATERIAL, GL20.GL_LINES).primitiveType should be (GL20.GL_LINES)

  "cube" should "be one model" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    Cube(ORIGIN, 1).getModel should have size 1

  it should "store one model" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    Cube.numStoredModels should be(1)

  "different cube models" should "be stored separately" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    Cube(ORIGIN, 1, primitiveType = GL20.GL_LINES).getModel
    Cube.numStoredModels should be(2)

  "cube toString" should "return class name" in:
    Cube().toString should be("Cube")

  "cube from squares" should "operate with six faces" taggedAs GdxTest in:
    assume(loadingLWJGLSucceeds)
    CubeFromSquares(ORIGIN, 1).getModel should have size 6

  "Composite with empty geometries list" should "return empty model list" taggedAs GdxTest in :
    assume(loadingLWJGLSucceeds)
    val composite = Composite(geometries = List.empty)
    composite.getModel should be(empty)
  
  "Composite with single geometry" should "return same models as the geometry" taggedAs GdxTest in :
    assume(loadingLWJGLSucceeds)
    val sphere = Sphere(ORIGIN, 1f)
    val composite = Composite(geometries = List(sphere))
    composite.getModel should have size sphere.getModel.size
  
  "Composite with multiple geometries" should "combine all models" taggedAs GdxTest in :
    assume(loadingLWJGLSucceeds)
    val sphere = Sphere(ORIGIN, 1f)
    val cube = Cube(ORIGIN, 1f)
    val composite = Composite(geometries = List(sphere, cube))
    val expectedSize = sphere.getModel.size + cube.getModel.size
    composite.getModel should have size expectedSize

  "Composite with nested composites" should "work correctly" taggedAs GdxTest in :
    assume(loadingLWJGLSucceeds)
    val sphere = Sphere(ORIGIN, 1f)
    val cube = Cube(ORIGIN, 1f)
    val innerComposite = Composite(geometries = List(sphere, cube))
    val square = Square(ORIGIN, 1f)
    val outerComposite = Composite(geometries = List(innerComposite, square))

    val expectedSize = sphere.getModel.size + cube.getModel.size + square.getModel.size
    outerComposite.getModel should have size expectedSize
