package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import menger.objects.FixedVector

/** A quadrilateral in 3D, unconstrained by the need to be rectilinear or parallel to any axis */
class Quad3D(v0: Vector3, v1: Vector3, v2: Vector3, v3: Vector3) extends FixedVector[4, Vector3](v0, v1, v2, v3)
