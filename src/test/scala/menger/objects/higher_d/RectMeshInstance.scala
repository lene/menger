package menger.objects.higher_d

import com.typesafe.scalalogging.LazyLogging
import com.badlogic.gdx.graphics.g3d.Material

class RectMeshInstance extends RectMesh with LazyLogging:
  override def modelPart(facesPart: Seq[QuadInfo], primitiveType: Int, material: Material): Unit = ()
