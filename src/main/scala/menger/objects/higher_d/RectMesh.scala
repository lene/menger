package menger.objects.higher_d

import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.{Material, Model}
import com.badlogic.gdx.graphics.g3d.utils.{MeshBuilder, MeshPartBuilder}
import com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo
import com.badlogic.gdx.math.{Vector3, Vector4}
import menger.objects.Builder

trait RectMesh:
  type RectVertices3D = (Vector3, Vector3, Vector3, Vector3)
  type RectInfo = (VertexInfo, VertexInfo, VertexInfo, VertexInfo)

  def model(faces: Seq[RectInfo], primitiveType: Int, material: Material): Model =
    Builder.modelBuilder.begin()
    faces.grouped(MeshBuilder.MAX_VERTICES / 4).foreach(facesPart =>
      val meshBuilder: MeshPartBuilder = Builder.modelBuilder.part(
        "sponge", primitiveType, Usage.Position | Usage.Normal, material
      )
      facesPart.foreach(face => meshBuilder.rect.tupled(face))
    )
    Builder.modelBuilder.end()

def edgeToString(edge: (Vector4, Vector4)): String = faceToString(Seq(edge._1, edge._2))
