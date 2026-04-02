package menger.objects

import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.Model
import com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder

trait ModelFactory:
  def createSphere(
    width: Float,
    height: Float,
    depth: Float,
    divisionsU: Int,
    divisionsV: Int,
    material: Material,
    attributes: Long
  ): Model

  def createBox(
    width: Float,
    height: Float,
    depth: Float,
    primitiveType: Int,
    material: Material,
    attributes: Long
  ): Model

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

  def begin(): Unit
  def part(id: String, primitiveType: Int, attributes: Long, material: Material): MeshPartBuilder
  def end(): Model

object ModelFactory:
  def default: ModelFactory = LibGDXModelFactory()

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
