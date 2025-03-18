package menger.objects.higher_d

import com.badlogic.gdx.graphics.g3d.{Material, Model}
import com.badlogic.gdx.graphics.g3d.utils.{MeshBuilder, MeshPartBuilder}
import com.badlogic.gdx.math.Vector4
import menger.objects.Builder

trait RectMesh:
  def model(faces: Seq[QuadInfo], primitiveType: Int, material: Material): Model =
    Builder.modelBuilder.begin()
    faces.grouped(MeshBuilder.MAX_VERTICES / 4).foreach(part => modelPart(part, primitiveType, material))
    Builder.modelBuilder.end()

  def modelPart(facesPart: Seq[QuadInfo], primitiveType: Int, material: Material): Unit =
    val meshBuilder = Builder.modelBuilder.part("sponge", primitiveType, Builder.DEFAULT_FLAGS, material)
    facesPart.foreach(face => meshBuilder.rect(face.v0, face.v1, face.v2, face.v3))
