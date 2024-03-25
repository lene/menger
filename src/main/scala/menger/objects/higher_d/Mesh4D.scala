package menger.objects.higher_d

trait Mesh4D extends RectMesh:
  lazy val faces: Seq[RectVertices4D]
