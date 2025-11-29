package menger.objects

/** Trait for objects that can produce renderer-agnostic triangle mesh data.
  *
  * Implemented by Face and other geometry-producing types to enable conversion to both LibGDX and
  * OptiX representations.
  */
trait TriangleMeshSource:
  def toTriangleMesh: TriangleMeshData
