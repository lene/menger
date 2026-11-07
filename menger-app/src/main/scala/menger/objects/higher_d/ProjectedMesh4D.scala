package menger.objects.higher_d

/** A 4D mesh drawn as a 3D triangle mesh: each vertex goes through `projectVertex`, the faces
  * are triangulated and the result is moved to `center`. */
trait ProjectedMesh4D extends ProjectedMesh[4]:
  def mesh4D: Mesh4D

  final def sourceMesh: Mesh[4] = mesh4D
