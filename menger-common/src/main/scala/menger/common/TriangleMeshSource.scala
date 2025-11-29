package menger.common

/** Trait for objects that can produce renderer-agnostic triangle mesh data.
  *
  * Implemented by geometry-producing types (Face, Cube, Sponge, etc.) to enable conversion to both
  * LibGDX and OptiX representations.
  */
trait TriangleMeshSource:
  def toTriangleMesh: TriangleMeshData
