package menger.objects

import com.badlogic.gdx.graphics.{GL20, Mesh}
import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder
import com.badlogic.gdx.graphics.g3d.{Material, ModelInstance}
import menger.objects.Direction.Z


class SpongeBySurface(
  level: Int, material: Material = Builder.WHITE_MATERIAL, primitiveType: Int = GL20.GL_TRIANGLES
) extends Geometry(material, primitiveType):

  override def at(x: Float, y: Float, z: Float, scale: Float): List[ModelInstance] =
    val faces = surfaces(Face(x, y, z, scale, Z))
    Builder.modelBuilder.begin()
    val meshBuilder: MeshPartBuilder = Builder.modelBuilder.part(
      "sponge", primitiveType, Usage.Position | Usage.Normal, material
    )
    faces.foreach(face =>
      meshBuilder.rect(face.vertices._1, face.vertices._2, face.vertices._3, face.vertices._4)
    )
    val mesh = Builder.modelBuilder.end()
    List(ModelInstance(mesh))

  private[objects] def surfaces(startFace: Face): Seq[Face] =
    val faces = Seq(startFace)
    level.until(0, -1).foldLeft(faces)(
      (faces, _) => faces.flatMap(_.subdivide())
    )
