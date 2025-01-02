package menger.objects.higher_d

import com.badlogic.gdx.graphics.VertexAttributes.Usage
import com.badlogic.gdx.graphics.g3d.{Material, Model}
import com.badlogic.gdx.graphics.g3d.utils.{MeshBuilder, MeshPartBuilder}
import com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo
import com.badlogic.gdx.math.{Vector3, Vector4}
import menger.objects.Builder

case class RectVertices4D(a: Vector4, b: Vector4, c: Vector4, d: Vector4):
  def asTuple: (Vector4, Vector4, Vector4, Vector4) = (a, b, c, d)
  def asSeq: Seq[Vector4] = Seq(a, b, c, d)
  def area: Float = (b - a).len() * (c - b).len()


object RectVertices4D:
  def apply(seq: Seq[Vector4]): RectVertices4D =
    require(seq.length == 4, s"Need 4 vertices, have ${seq.length}: ${seq.map(vec2string)}")
    RectVertices4D(seq.head, seq(1), seq(2), seq(3))

trait RectMesh:
  type RectIndices = (Int, Int, Int, Int)
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

def faceToString(face: Seq[Vector4]): String = face.map(vec2string).mkString("(", ", ", ")")
def faceToString(face: RectVertices4D): String = faceToString(face.asSeq)
def faceToString(face: (Vector4, Vector4, Vector4, Vector4)): String =
  faceToString(Seq(face._1, face._2, face._3, face._4))
def edgeToString(edge: (Vector4, Vector4)): String = faceToString(Seq(edge._1, edge._2))
