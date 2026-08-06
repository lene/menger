package menger.input

import com.badlogic.gdx.math.{Vector3 => GdxVector3}
import menger.common.Vector

/** GDX Vector3 <-> menger.common.Vector[3] bridge, confined to the input/engines
  * boundary where LibGDX's orbit-camera math meets the common-typed OptiX renderer API. */
object Vector3Extensions:
  extension (v: GdxVector3)
    def toVector3: Vector[3] = Vector[3](v.x, v.y, v.z)

  extension (v: Vector[3])
    def toGdxVector3: GdxVector3 = GdxVector3(v(0), v(1), v(2))
