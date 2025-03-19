package menger.objects.higher_d

import com.badlogic.gdx.graphics.g3d.{Material, Model}
import com.badlogic.gdx.graphics.g3d.utils.{MeshBuilder, MeshPartBuilder}
import menger.objects.Builder

trait RectMesh:
  def model(faces: Seq[QuadInfo], primitiveType: Int, material: Material): Model =
    Builder.modelBuilder.begin()
    faces.grouped(MeshBuilder.MAX_VERTICES / 4).foreach(facesPart =>
      val meshBuilder: MeshPartBuilder = Builder.modelBuilder.part(
        "sponge", primitiveType, Builder.DEFAULT_FLAGS, material
      )
      facesPart.foreach(face => meshBuilder.rect(face.v0, face.v1, face.v2, face.v3))
    )
    Builder.modelBuilder.end()
