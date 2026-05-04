package menger.objects.higher_d

trait Mesh4D extends RectMesh:
  type V <: Int & Singleton
  lazy val faces: Seq[Face4D[V]]
  def vertsPerFace: Int = faces.headOption.map(_.vertsPerFace).getOrElse(4)

  override def toString: String = s"${getClass.getSimpleName}(${faces.size} faces)"
