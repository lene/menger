package menger.objects

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ParametricTessellatorSuite extends AnyFlatSpec with Matchers:

  private val flatPlane: (Float, Float) => (Float, Float, Float) =
    (u, v) => (u, 0f, v)

  "ParametricTessellator" should "produce correct vertex count for open surface" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = false, closedV = false
    )
    mesh.numVertices shouldBe (4 + 1) * (4 + 1)  // 25

  it should "produce correct triangle count" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = false, closedV = false
    )
    mesh.numTriangles shouldBe 2 * 4 * 4  // 32

  it should "use stride 8 (pos + normal + uv)" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = false, closedV = false
    )
    mesh.vertexStride shouldBe 8

  it should "compute normals approximately (0,1,0) or (0,-1,0) for flat plane" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = false, closedV = false
    )
    for i <- 0 until mesh.numVertices do
      val nx = mesh.vertices(i * 8 + 3)
      val ny = mesh.vertices(i * 8 + 4)
      val nz = mesh.vertices(i * 8 + 5)
      // Normal should be (0, +-1, 0) for flat xz plane
      math.abs(nx) should be < 0.01f
      math.abs(ny) should be > 0.99f
      math.abs(nz) should be < 0.01f
