package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.utils.{MeshBuilder, MeshPartBuilder}
import com.badlogic.gdx.graphics.g3d.{Material, Model, ModelInstance}
import com.badlogic.gdx.math.Vector3
import menger.objects.Direction.Z


class SpongeBySurface(
  level: Int, material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(material, primitiveType):
  require(level >= 0, "Level must be non-negative")

  override def at(center: Vector3, scale: Float): List[ModelInstance] = logTime("at", 5) {
    val facingPlusX = transformed(ModelInstance(mesh), scale, center, 0, 1, 0, 90)
    val facingMinusX = transformed(ModelInstance(mesh), scale, center, 0, 1, 0, -90)
    val facingPlusY = transformed(ModelInstance(mesh), scale, center, 1, 0, 0, 90)
    val facingMinusY = transformed(ModelInstance(mesh), scale, center, 1, 0, 0, -90)
    val facingPlusZ = transformed(ModelInstance(mesh), scale, center, 0, 1, 0, 0)
    val facingMinusZ = transformed(ModelInstance(mesh), scale, center, 0, 1, 0, 180)

    List(facingPlusX, facingMinusX, facingPlusY, facingMinusY, facingPlusZ, facingMinusZ)
  }

  private def transformed(
    modelInstance: ModelInstance, scale: Float, xlate: Vector3, axisX: Float, axisY: Float, axisZ: Float, angle: Float
  ): ModelInstance =
    modelInstance.transform.translate(xlate)
    modelInstance.transform.rotate(axisX, axisY, axisZ, angle)
    modelInstance.transform.translate(0, 0, scale / 2)
    modelInstance.transform.scale(scale, scale, scale)
    modelInstance

  override def toString: String = s"SpongeBySurface(level=$level, ${6 * faces.size} faces)"

  private[objects] def surfaces(startFace: Face): Seq[Face] =
    val faces = Seq(startFace)
    level.until(0, -1).foldLeft(faces)(
      (faces, _) => faces.flatMap(_.subdivide())
    )

  lazy val faces: Seq[Face] = logTime("faces", 5) { surfaces(Face(0, 0, 0, 1, Z)) }

  lazy val mesh: Model = logTime("mesh", 5) {
      Builder.modelBuilder.begin()
      faces.grouped(MeshBuilder.MAX_VERTICES / 4).foreach(facesPart =>
        val meshBuilder = Builder.modelBuilder.part("sponge", primitiveType, Builder.DEFAULT_FLAGS, material)
        facesPart.foreach(face => meshBuilder.rect.tupled(face.vertices))
      )
      Builder.modelBuilder.end()
    }
