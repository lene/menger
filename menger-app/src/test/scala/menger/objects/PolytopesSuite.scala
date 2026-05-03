package menger.objects

import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PolytopesSuite extends AnyFlatSpec with Matchers:

  private val Eps = 1e-4f

  private def norm(x: Float, y: Float, z: Float): Float =
    math.sqrt(x * x + y * y + z * z).toFloat

  private def allVerticesOnUnitSphere(data: TriangleMeshData): Unit =
    val stride = data.vertexStride
    for i <- 0 until data.numVertices do
      val base = i * stride
      val d = norm(data.vertices(base), data.vertices(base + 1), data.vertices(base + 2))
      d shouldBe (1f +- Eps)

  private def hasPositiveOutwardNormals(data: TriangleMeshData): Unit =
    val v = data.vertices
    val stride = data.vertexStride
    val idx = data.indices
    for i <- idx.indices by 3 do
      val (i0, i1, i2) = (idx(i) * stride, idx(i + 1) * stride, idx(i + 2) * stride)
      val (ax, ay, az) = (v(i0), v(i0 + 1), v(i0 + 2))
      val (bx, by, bz) = (v(i1), v(i1 + 1), v(i1 + 2))
      val (cx, cy, cz) = (v(i2), v(i2 + 1), v(i2 + 2))
      val (e1x, e1y, e1z) = (bx - ax, by - ay, bz - az)
      val (e2x, e2y, e2z) = (cx - ax, cy - ay, cz - az)
      val (nx, ny, nz) = (
        e1y * e2z - e1z * e2y,
        e1z * e2x - e1x * e2z,
        e1x * e2y - e1y * e2x
      )
      val (centX, centY, centZ) = ((ax + bx + cx) / 3f, (ay + by + cy) / 3f, (az + bz + cz) / 3f)
      val dot = nx * centX + ny * centY + nz * centZ
      dot should be > 0f

  "Tetrahedron" should "have 4 vertices" in:
    val data = Tetrahedron(Vector3(0f, 0f, 0f), 1f).toTriangleMesh
    data.numVertices shouldBe 4

  it should "have 4 triangles (12 indices)" in:
    val data = Tetrahedron(Vector3(0f, 0f, 0f), 1f).toTriangleMesh
    data.indices.length shouldBe 12

  it should "have all vertices on unit sphere" in:
    val data = Tetrahedron(Vector3(0f, 0f, 0f), 1f).toTriangleMesh
    allVerticesOnUnitSphere(data)

  it should "have outward face normals" in:
    val data = Tetrahedron(Vector3(0f, 0f, 0f), 1f).toTriangleMesh
    hasPositiveOutwardNormals(data)

  it should "scale vertices by scale parameter" in:
    val data = Tetrahedron(Vector3(0f, 0f, 0f), 2f).toTriangleMesh
    val stride = data.vertexStride
    for i <- 0 until data.numVertices do
      val base = i * stride
      val d = norm(data.vertices(base), data.vertices(base + 1), data.vertices(base + 2))
      d shouldBe (2f +- Eps)

  it should "translate vertices by center parameter" in:
    val center = Vector3(1f, 2f, 3f)
    val data = Tetrahedron(center, 1f).toTriangleMesh
    val coords = data.vertices
    val stride = TriangleMeshData.LegacyVertexStride
    for i <- coords.indices by stride do
      val d = norm(coords(i) - 1f, coords(i + 1) - 2f, coords(i + 2) - 3f)
      d shouldBe (1f +- Eps)

  @SuppressWarnings(Array("org.wartremover.warts.Var"))
  private def hasVertexAtUnitDistanceFromCenter(data: TriangleMeshData, center: Vector3): Unit =
    val stride = data.vertexStride
    var found = false
    for i <- 0 until data.numVertices do
      val base = i * stride
      val d = norm(data.vertices(base) - center.x, data.vertices(base + 1) - center.y, data.vertices(base + 2) - center.z)
      if math.abs(d - 1f) < Eps then found = true
    found shouldBe true

  "Octahedron" should "have 6 vertices" in:
    val data = Octahedron(Vector3(0f, 0f, 0f), 1f).toTriangleMesh
    data.vertices.length / TriangleMeshData.LegacyVertexStride shouldBe 6

  it should "have 8 triangles (24 indices)" in:
    val data = Octahedron(Vector3(0f, 0f, 0f), 1f).toTriangleMesh
    data.indices.length shouldBe 24

  it should "have all vertices on unit sphere" in:
    val data = Octahedron(Vector3(0f, 0f, 0f), 1f).toTriangleMesh
    allVerticesOnUnitSphere(data)

  it should "have outward face normals" in:
    val data = Octahedron(Vector3(0f, 0f, 0f), 1f).toTriangleMesh
    hasPositiveOutwardNormals(data)

  it should "scale vertices by scale parameter" in:
    val data = Octahedron(Vector3(0f, 0f, 0f), 2f).toTriangleMesh
    val coords = data.vertices
    val stride = TriangleMeshData.LegacyVertexStride
    for i <- coords.indices by stride do
      val d = norm(coords(i), coords(i + 1), coords(i + 2))
      d shouldBe (2f +- Eps)

  it should "translate vertices by center parameter" in:
    val center = Vector3(1f, 2f, 3f)
    val data = Octahedron(center, 1f).toTriangleMesh
    val coords = data.vertices
    val stride = TriangleMeshData.LegacyVertexStride
    for i <- coords.indices by stride do
      val d = norm(coords(i) - 1f, coords(i + 1) - 2f, coords(i + 2) - 3f)
      d shouldBe (1f +- Eps)

  "Icosahedron" should "have 12 vertices" in:
    val data = Icosahedron(Vector3(0f, 0f, 0f), 1f).toTriangleMesh
    data.vertices.length / TriangleMeshData.LegacyVertexStride shouldBe 12

  it should "have 20 triangles (60 indices)" in:
    val data = Icosahedron(Vector3(0f, 0f, 0f), 1f).toTriangleMesh
    data.indices.length shouldBe 60

  it should "have all vertices on unit sphere" in:
    val data = Icosahedron(Vector3(0f, 0f, 0f), 1f).toTriangleMesh
    allVerticesOnUnitSphere(data)

  it should "have outward face normals" in:
    val data = Icosahedron(Vector3(0f, 0f, 0f), 1f).toTriangleMesh
    hasPositiveOutwardNormals(data)

  it should "scale vertices by scale parameter" in:
    val data = Icosahedron(Vector3(0f, 0f, 0f), 2f).toTriangleMesh
    val coords = data.vertices
    val stride = TriangleMeshData.LegacyVertexStride
    for i <- coords.indices by stride do
      val d = norm(coords(i), coords(i + 1), coords(i + 2))
      d shouldBe (2f +- Eps)

  it should "translate vertices by center parameter" in:
    val center = Vector3(1f, 2f, 3f)
    val data = Icosahedron(center, 1f).toTriangleMesh
    val coords = data.vertices
    val stride = TriangleMeshData.LegacyVertexStride
    for i <- coords.indices by stride do
      val d = norm(coords(i) - 1f, coords(i + 1) - 2f, coords(i + 2) - 3f)
      d shouldBe (1f +- Eps)
