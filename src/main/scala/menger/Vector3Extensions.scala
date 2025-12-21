package menger

import com.badlogic.gdx.math.Vector3
import menger.common.Vector

object Vector3Extensions:
  extension (v: Vector3)
    def toVector3: Vector[3] = Vector[3](v.x, v.y, v.z)
