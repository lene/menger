package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3

object VertexInfo:
  def apply(v: Vector3): com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo =
    com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo().setPos(v)
