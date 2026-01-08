package menger.objects.higher_d

import com.badlogic.gdx.graphics.g3d.Material
import com.typesafe.scalalogging.LazyLogging

class RectMeshInstance extends RectMesh with LazyLogging:
  override def modelPart(facesPart: Seq[QuadInfo], primitiveType: Int, material: Material): Unit = ()
