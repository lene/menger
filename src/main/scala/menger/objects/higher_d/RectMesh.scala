package menger.objects.higher_d

import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.{Material, Model}
import com.badlogic.gdx.graphics.g3d.utils.{MeshBuilder, MeshPartBuilder}
import com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo
import com.badlogic.gdx.math.{Vector3, Vector4}
import menger.objects.Builder

trait RectMesh:
  type RectIndices = (Int, Int, Int, Int)
  type RectVertices4D = (Vector4, Vector4, Vector4, Vector4)
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

def faceToString(face: RectMesh#RectVertices4D): String =
  s"(${vec2string(face._1)}, ${vec2string(face._2)}, ${vec2string(face._3)}, ${vec2string(face._4)})"

def area(face: RectMesh#RectVertices4D): Float =
  val (a, b, c, _) = face
  (b - a).len() * (c - b).len()
