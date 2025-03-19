package menger.objects.higher_d

import com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo

class QuadInfo(val v0: VertexInfo, val v1: VertexInfo, val v2: VertexInfo, val v3: VertexInfo)
  extends FixedVector[4, VertexInfo](v0, v1, v2, v3)

