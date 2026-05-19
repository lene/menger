package menger.objects.higher_d

import menger.common.Const
import menger.common.Vector

object CellShapeContract:

  type Cell4D = Seq[Vector[4]]

  private def minEdgeLen(cell: Cell4D): Float =
    (for i <- cell.indices; j <- cell.indices if i < j yield cell(i).dst(cell(j))).min

  private def edgeCount(cell: Cell4D, edgeLen: Float): Int =
    val tol = math.max(Const.epsilon * 100, edgeLen * 0.01f)
    (for i <- cell.indices; j <- cell.indices if i < j
     yield math.abs(cell(i).dst(cell(j)) - edgeLen) < tol).count(identity)

  def assertTetrahedron(cell: Cell4D): Unit =
    assert(cell.size == 4, s"Tetrahedron: need 4 vertices, got ${cell.size}")
    val e = minEdgeLen(cell)
    assert(edgeCount(cell, e) == 6, s"Tetrahedron: need 6 equal edges at len $e, got ${edgeCount(cell, e)}")

  def assertOctahedron(cell: Cell4D): Unit =
    assert(cell.size == 6, s"Octahedron: need 6 vertices, got ${cell.size}")
    val e = minEdgeLen(cell)
    assert(edgeCount(cell, e) == 12, s"Octahedron: need 12 equal edges at len $e, got ${edgeCount(cell, e)}")

  def assertCube(cell: Cell4D): Unit =
    assert(cell.size == 8, s"Cube: need 8 vertices, got ${cell.size}")
    val e = minEdgeLen(cell)
    assert(edgeCount(cell, e) == 12, s"Cube: need 12 equal edges at len $e, got ${edgeCount(cell, e)}")

  def assertDodecahedron(cell: Cell4D): Unit =
    assert(cell.size == 20, s"Dodecahedron: need 20 vertices, got ${cell.size}")
    val e = minEdgeLen(cell)
    assert(edgeCount(cell, e) == 30, s"Dodecahedron: need 30 equal edges at len $e, got ${edgeCount(cell, e)}")
