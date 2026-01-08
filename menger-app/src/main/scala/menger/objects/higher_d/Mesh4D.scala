package menger.objects.higher_d

trait Mesh4D extends RectMesh:
  lazy val faces: Seq[Face4D]

  override def toString: String = s"${getClass.getSimpleName}(${faces.size} faces)"
