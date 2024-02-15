package menger.objects

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.{GL20, Mesh}
import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.utils.{MeshBuilder, MeshPartBuilder}
import com.badlogic.gdx.graphics.g3d.{Material, Model, ModelInstance}
import menger.objects.Direction.Z


class SpongeBySurface(
  level: Int, material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(material, primitiveType):

  override def at(x: Float, y: Float, z: Float, scale: Float): List[ModelInstance] =
    logTime("at") {
      val facingPlusX = ModelInstance(mesh)
      facingPlusX.transform.rotate(0, 1, 0, 90)
      facingPlusX.transform.translate(0, 0, scale / 2)
      val facingMinusX = ModelInstance(mesh)
      facingMinusX.transform.rotate(0, 1, 0, -90)
      facingMinusX.transform.translate(0, 0, scale / 2)

      val facingPlusY = ModelInstance(mesh)
      facingPlusY.transform.rotate(1, 0, 0, 90)
      facingPlusY.transform.translate(0, 0, scale / 2)
      val facingMinusY = ModelInstance(mesh)
      facingMinusY.transform.rotate(1, 0, 0, -90)
      facingMinusY.transform.translate(0, 0, scale / 2)

      val facingPlusZ = ModelInstance(mesh)
      facingPlusZ.transform.translate(0, 0, scale / 2)
      val facingMinusZ = ModelInstance(mesh)
      facingMinusZ.transform.rotate(0, 1, 0, 180)
      facingMinusZ.transform.translate(0, 0, scale / 2)

      List(facingPlusX, facingMinusX, facingPlusY, facingMinusY, facingPlusZ, facingMinusZ)
    }

  override def toString: String = s"SpongeBySurface(level=$level, ${6 * faces.size} faces)"

  private[objects] def surfaces(startFace: Face): Seq[Face] =
    val faces = Seq(startFace)
    level.until(0, -1).foldLeft(faces)(
      (faces, _) => faces.flatMap(_.subdivide())
    )

  lazy val faces: Seq[Face] = logTime("faces") {
    surfaces(Face(0, 0, 0, 1, Z))
  }
  lazy val mesh: Model =
    if level < 0 then throw new IllegalArgumentException("Level must be >= 0")
    logTime("mesh") {
      Builder.modelBuilder.begin()
      faces.grouped(MeshBuilder.MAX_VERTICES / 4).foreach(facesPart =>
        val meshBuilder: MeshPartBuilder = Builder.modelBuilder.part(
          "sponge", primitiveType, Usage.Position | Usage.Normal, material
        )
        facesPart.foreach(face => meshBuilder.rect.tupled(face.vertices))
      )
      Builder.modelBuilder.end()
    }
