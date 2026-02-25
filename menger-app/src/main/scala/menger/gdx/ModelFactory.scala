package menger.gdx

import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.Model
import com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder

/**
 * Abstraction for creating LibGDX 3D models.
 *
 * This trait isolates model creation from the rest of the application, enabling:
 * - Testing without LibGDX initialization
 * - Mocking model creation in unit tests
 * - Dependency injection of model factory implementations
 *
 * Part of the menger.gdx wrapper layer (Task 11.1).
 */
trait ModelFactory:
  /**
   * Create a sphere model.
   *
   * @param width Width of the sphere
   * @param height Height of the sphere
   * @param depth Depth of the sphere
   * @param divisionsU Number of horizontal divisions
   * @param divisionsV Number of vertical divisions
   * @param material Material for the sphere
   * @param attributes Vertex attributes flags
   * @return The created model
   */
  def createSphere(
    width: Float,
    height: Float,
    depth: Float,
    divisionsU: Int,
    divisionsV: Int,
    material: Material,
    attributes: Long
  ): Model

  /**
   * Create a box model.
   *
   * @param width Width of the box
   * @param height Height of the box
   * @param depth Depth of the box
   * @param primitiveType OpenGL primitive type (e.g., GL_TRIANGLES, GL_LINES)
   * @param material Material for the box
   * @param attributes Vertex attributes flags
   * @return The created model
   */
  def createBox(
    width: Float,
    height: Float,
    depth: Float,
    primitiveType: Int,
    material: Material,
    attributes: Long
  ): Model

  /**
   * Create a rectangle (quad) model.
   *
   * @param x00 X coordinate of corner 0
   * @param y00 Y coordinate of corner 0
   * @param z00 Z coordinate of corner 0
   * @param x10 X coordinate of corner 1
   * @param y10 Y coordinate of corner 1
   * @param z10 Z coordinate of corner 1
   * @param x11 X coordinate of corner 2
   * @param y11 Y coordinate of corner 2
   * @param z11 Z coordinate of corner 2
   * @param x01 X coordinate of corner 3
   * @param y01 Y coordinate of corner 3
   * @param z01 Z coordinate of corner 3
   * @param normalX X component of normal vector
   * @param normalY Y component of normal vector
   * @param normalZ Z component of normal vector
   * @param primitiveType OpenGL primitive type (e.g., GL_TRIANGLES)
   * @param material Material for the rectangle
   * @param attributes Vertex attributes flags
   * @return The created model
   */
  def createRect(
    x00: Float, y00: Float, z00: Float,
    x10: Float, y10: Float, z10: Float,
    x11: Float, y11: Float, z11: Float,
    x01: Float, y01: Float, z01: Float,
    normalX: Float, normalY: Float, normalZ: Float,
    primitiveType: Int,
    material: Material,
    attributes: Long
  ): Model

  /**
   * Begin building a complex model with multiple parts.
   *
   * Call `part()` to add mesh parts, then `end()` to finalize.
   */
  def begin(): Unit

  /**
   * Add a mesh part to the model being built.
   *
   * @param id Identifier for this part
   * @param primitiveType OpenGL primitive type (e.g., GL_TRIANGLES)
   * @param attributes Vertex attributes flags
   * @param material Material for this part
   * @return MeshPartBuilder for adding geometry to this part
   */
  def part(id: String, primitiveType: Int, attributes: Long, material: Material): MeshPartBuilder

  /**
   * Finalize and return the complex model being built.
   *
   * @return The completed model
   */
  def end(): Model

object ModelFactory:
  /**
   * Default ModelFactory implementation using LibGDX ModelBuilder.
   *
   * Requires LibGDX/LWJGL initialization.
   */
  def default: ModelFactory = LibGDXModelFactory()

/**
 * Production ModelFactory implementation using LibGDX ModelBuilder.
 */
private class LibGDXModelFactory extends ModelFactory:
  private val builder = ModelBuilder()

  def createSphere(
    width: Float,
    height: Float,
    depth: Float,
    divisionsU: Int,
    divisionsV: Int,
    material: Material,
    attributes: Long
  ): Model =
    builder.createSphere(width, height, depth, divisionsU, divisionsV, material, attributes)

  def createBox(
    width: Float,
    height: Float,
    depth: Float,
    primitiveType: Int,
    material: Material,
    attributes: Long
  ): Model =
    builder.createBox(width, height, depth, primitiveType, material, attributes)

  def createRect(
    x00: Float, y00: Float, z00: Float,
    x10: Float, y10: Float, z10: Float,
    x11: Float, y11: Float, z11: Float,
    x01: Float, y01: Float, z01: Float,
    normalX: Float, normalY: Float, normalZ: Float,
    primitiveType: Int,
    material: Material,
    attributes: Long
  ): Model =
    builder.createRect(
      x00, y00, z00,
      x10, y10, z10,
      x11, y11, z11,
      x01, y01, z01,
      normalX, normalY, normalZ,
      primitiveType, material, attributes
    )

  def begin(): Unit = builder.begin()

  def part(id: String, primitiveType: Int, attributes: Long, material: Material): MeshPartBuilder =
    builder.part(id, primitiveType, attributes, material)

  def end(): Model = builder.end()
