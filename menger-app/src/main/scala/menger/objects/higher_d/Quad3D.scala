package menger.objects.higher_d

import menger.common.Vector
import menger.objects.FixedVector


class Quad3D(v0: Vector[3], v1: Vector[3], v2: Vector[3], v3: Vector[3]) extends FixedVector[4, Vector[3]](v0, v1, v2, v3)
