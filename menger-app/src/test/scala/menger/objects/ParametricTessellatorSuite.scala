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

  it should "reduce vertex count for closedU surface" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = true, closedV = false
    )
    // closedU: uVerts = uSteps (4), vVerts = vSteps+1 (5) => 20
    mesh.numVertices shouldBe 4 * 5

  it should "reduce vertex count for closedU + closedV surface" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = true, closedV = true
    )
    mesh.numVertices shouldBe 4 * 4  // 16

  it should "still produce same triangle count for closed surface" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = true, closedV = true
    )
    mesh.numTriangles shouldBe 2 * 4 * 4

  it should "produce valid mesh with minimal resolution (1x1)" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 1, 1, closedU = false, closedV = false
    )
    mesh.numVertices shouldBe 4
    mesh.numTriangles shouldBe 2

  it should "compute accurate normals for tessellated sphere" in:
    import scala.math._
    val sphereF: (Float, Float) => (Float, Float, Float) = (u, v) =>
      (cos(u).toFloat * sin(v).toFloat, cos(v).toFloat, sin(u).toFloat * sin(v).toFloat)

    val mesh = ParametricTessellator.tessellate(
      sphereF, (0f, (2 * Pi).toFloat), (0.1f, (Pi - 0.1).toFloat),
      64, 32, closedU = true, closedV = false
    )
    // Check normals match normalized position (sphere property)
    for i <- 0 until mesh.numVertices do
      val base = i * 8
      val px = mesh.vertices(base); val py = mesh.vertices(base + 1); val pz = mesh.vertices(base + 2)
      val nx = mesh.vertices(base + 3); val ny = mesh.vertices(base + 4); val nz = mesh.vertices(base + 5)
      val pLen = sqrt(px * px + py * py + pz * pz).toFloat
      if pLen > 0.01f then
        val epx = px / pLen; val epy = py / pLen; val epz = pz / pLen
        // Dot product should be close to 1 (or -1 if flipped)
        val dot = math.abs(nx * epx + ny * epy + nz * epz)
        dot should be > 0.95f  // within ~18 degrees

  it should "produce UV coordinates in [0, 1]" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (-2f, 2f), (-3f, 3f), 8, 8, closedU = false, closedV = false
    )
    for i <- 0 until mesh.numVertices do
      val texU = mesh.vertices(i * 8 + 6)
      val texV = mesh.vertices(i * 8 + 7)
      texU should be >= 0f
      texU should be <= 1f
      texV should be >= 0f
      texV should be <= 1f

  it should "have consistent triangle winding order" in:
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 4, 4, closedU = false, closedV = false
    )
    // For a flat xz plane at y=0, compute the y-component of each triangle's
    // edge cross product and verify they're all the same sign.
    val crossYValues = (0 until mesh.numTriangles).map: t =>
      val i0 = mesh.indices(t * 3); val i1 = mesh.indices(t * 3 + 1); val i2 = mesh.indices(t * 3 + 2)
      val ax = mesh.vertices(i1 * 8) - mesh.vertices(i0 * 8)
      val az = mesh.vertices(i1 * 8 + 2) - mesh.vertices(i0 * 8 + 2)
      val bx = mesh.vertices(i2 * 8) - mesh.vertices(i0 * 8)
      val bz = mesh.vertices(i2 * 8 + 2) - mesh.vertices(i0 * 8 + 2)
      ax * bz - az * bx
    // All triangles should wind the same way
    (crossYValues.forall(_ > 0) || crossYValues.forall(_ < 0)) shouldBe true

  it should "accept resolution well below warning threshold without crashing" in:
    // The memory warning fires at uSteps*vSteps > 1,000,000. This test uses a small grid
    // to verify basic operation; the warning itself is verified visually at WARN level.
    val mesh = ParametricTessellator.tessellate(
      flatPlane, (0f, 1f), (0f, 1f), 2, 2, closedU = false, closedV = false
    )
    mesh.numVertices shouldBe 9

  "ObjectType" should "recognize 'parametric' as valid" in:
    menger.common.ObjectType.isValid("parametric") shouldBe true

  it should "not classify 'parametric' as sponge or 4D" in:
    menger.common.ObjectType.isSponge("parametric") shouldBe false
    menger.common.ObjectType.isProjected4D("parametric") shouldBe false

  "ParametricSurface" should "produce an ObjectSpec with meshData" in:
    import menger.dsl._
    val surface = ParametricSurface(
      f = (u, v) => Vec3(u, 0f, v),
      uRange = (0f, 1f),
      vRange = (0f, 1f),
      uSteps = 4,
      vSteps = 4
    )
    val spec = surface.toObjectSpec
    spec.objectType shouldBe "parametric"
    spec.meshData shouldBe defined
    spec.meshData.get.numVertices shouldBe 25
    spec.meshData.get.numTriangles shouldBe 32

  "ParametricTessellator" should "handle degenerate normals at sphere poles" in:
    import scala.math._
    val sphereF: (Float, Float) => (Float, Float, Float) = (u, v) =>
      (cos(u).toFloat * sin(v).toFloat, cos(v).toFloat, sin(u).toFloat * sin(v).toFloat)

    // Include poles (v=0 and v=Pi)
    val mesh = ParametricTessellator.tessellate(
      sphereF, (0f, (2 * Pi).toFloat), (0f, Pi.toFloat),
      16, 8, closedU = true, closedV = false
    )
    // No NaN normals
    for i <- 0 until mesh.numVertices do
      val nx = mesh.vertices(i * 8 + 3)
      val ny = mesh.vertices(i * 8 + 4)
      val nz = mesh.vertices(i * 8 + 5)
      nx.isNaN shouldBe false
      ny.isNaN shouldBe false
      nz.isNaN shouldBe false
      val len = math.sqrt(nx * nx + ny * ny + nz * nz).toFloat
      len should be > 0.99f
      len should be < 1.01f
