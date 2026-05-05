package menger.objects.higher_d

trait Mesh4D extends RectMesh:
  type V <: Int & Singleton
  type Cell4D = Seq[menger.common.Vector[4]]

  lazy val faces: Seq[Face4D[V]]
  def vertsPerFace: Int = faces.headOption.map(_.vertsPerFace).getOrElse(4)

  lazy val edges: Set[(menger.common.Vector[4], menger.common.Vector[4])] =
    type VKey = (Float, Float, Float, Float)
    def vkey(v: menger.common.Vector[4]): VKey = (v(0), v(1), v(2), v(3))
    def canonical(a: menger.common.Vector[4], b: menger.common.Vector[4]): (menger.common.Vector[4], menger.common.Vector[4]) =
      if summon[Ordering[VKey]].lteq(vkey(a), vkey(b)) then (a, b) else (b, a)
    faces.flatMap { f =>
      (0 until f.vertsPerFace).map { i =>
        canonical(f(i), f((i + 1) % f.vertsPerFace))
      }
    }.toSet

  def cells: Seq[Cell4D] = Seq.empty

  override def toString: String = s"${getClass.getSimpleName}(${faces.size} faces)"
