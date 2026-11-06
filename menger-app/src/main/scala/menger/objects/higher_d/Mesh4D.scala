package menger.objects.higher_d

import menger.common.Vector

trait Mesh[N <: Int & Singleton]:
  type V <: Int & Singleton
  type Cell = Seq[Vector[N]]

  def vertices: Seq[Vector[N]]
  lazy val faces: Seq[Polygon[N, V]]
  def vertsPerFace: Int = faces.headOption.map(_.vertsPerFace).getOrElse(4)

  lazy val edges: Set[(Vector[N], Vector[N])] =
    faces.flatMap { f =>
      (0 until f.vertsPerFace).map { i =>
        Mesh.canonical(f(i), f((i + 1) % f.vertsPerFace))
      }
    }.toSet

  def cells: Seq[Cell] = Seq.empty

  override def toString: String = s"${getClass.getSimpleName}(${faces.size} faces)"

object Mesh:
  private val floatOrdering = summon[Ordering[Float]]

  private def lexicographicallyAtMost[N <: Int & Singleton](a: Vector[N], b: Vector[N]): Boolean =
    a.toIndexedSeq.zip(b.toIndexedSeq)
      .map((x, y) => floatOrdering.compare(x, y))
      .find(_ != 0)
      .forall(_ < 0)

  private def canonical[N <: Int & Singleton](
    a: Vector[N], b: Vector[N]
  ): (Vector[N], Vector[N]) =
    if lexicographicallyAtMost(a, b) then (a, b) else (b, a)

type Mesh4D = Mesh[4]
type Cell4D = Seq[Vector[4]]
