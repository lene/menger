package menger.objects

import menger.common.Vector
import menger.common.x
import menger.common.y
import menger.common.z
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.tagobjects.Slow

class SpongeByVolumeMeshSuite extends AnyFlatSpec with Matchers:

  "Cube.toTriangleMesh" should "generate a cube mesh with 12 triangles" in:
    val cube = Cube(Vector.Zero[3], 1.0f)
    val mesh = cube.toTriangleMesh
    mesh.numTriangles shouldBe 12
    mesh.numVertices shouldBe 24

  "SpongeByVolume.toTriangleMesh" should "generate a single cube at level 0" in:
    val sponge = SpongeByVolume(Vector.Zero[3], 1.0f, level = 0)
    val mesh = sponge.toTriangleMesh
    mesh.numTriangles shouldBe 12
    mesh.numVertices shouldBe 24

  it should "generate exterior-only faces at level 1 (144 triangles after culling interior shared faces)" in:
    val sponge = SpongeByVolume(Vector.Zero[3], 1.0f, level = 1)
    val mesh = sponge.toTriangleMesh
    mesh.numTriangles shouldBe 144
    mesh.numVertices shouldBe 288

  it should "generate exterior-only faces at level 2 (2112 triangles after culling interior shared faces)" taggedAs Slow in:
    val sponge = SpongeByVolume(Vector.Zero[3], 1.0f, level = 2)
    val mesh = sponge.toTriangleMesh
    mesh.numTriangles shouldBe 2112
    mesh.numVertices shouldBe 4224

  it should "position cubes correctly at level 1" in:
    val sponge = SpongeByVolume(Vector.Zero[3], 3.0f, level = 1)
    val mesh = sponge.toTriangleMesh
    val stride = mesh.vertexStride

    val vertices = mesh.vertices
    val xs = (0 until mesh.numVertices).map(i => vertices(i * stride))
    val ys = (0 until mesh.numVertices).map(i => vertices(i * stride + 1))
    val zs = (0 until mesh.numVertices).map(i => vertices(i * stride + 2))

    xs.min should be < -0.9f
    xs.max should be > 0.9f
    ys.min should be < -0.9f
    ys.max should be > 0.9f
    zs.min should be < -0.9f
    zs.max should be > 0.9f

  it should "preserve proper normals for faces" in:
    val sponge = SpongeByVolume(Vector.Zero[3], 1.0f, level = 0)
    val mesh = sponge.toTriangleMesh
    val stride = mesh.vertexStride

    for i <- 0 until mesh.numVertices do
      val nx = mesh.vertices(i * stride + 3)
      val ny = mesh.vertices(i * stride + 4)
      val nz = mesh.vertices(i * stride + 5)
      val lengthSquared = nx * nx + ny * ny + nz * nz
      lengthSquared shouldBe 1.0f +- 0.01f

  it should "generate a mesh at fractional level 0.5" in:
    val sponge = SpongeByVolume(Vector.Zero[3], 1.0f, level = 0.5f)
    val mesh = sponge.toTriangleMesh
    mesh.numTriangles should be > 0

  it should "generate a mesh at fractional level 1.5" in:
    val sponge = SpongeByVolume(Vector.Zero[3], 1.0f, level = 1.5f)
    val mesh = sponge.toTriangleMesh
    mesh.numTriangles should be > 0

  // Usability review 2026-10, session 4 (F86): the generator was rewritten for speed (one array
  // instead of a merge per sub-cube per level). It must produce the same mesh, byte for byte,
  // as the recursive merge it replaced -- kept below as the reference.
  it should "produce exactly the mesh of the recursive per-level merge it replaced" in:
    val centre = Vector[3](0.5f, -1f, 2f)
    for
      level <- 0 to 3
      excluded <- Seq(Set.empty[Direction], Set(Direction.X, Direction.negZ))
    do
      val mesh = SpongeByVolume(centre, 2.5f, level.toFloat).toTriangleMeshExcluding(excluded)
      val reference = legacyMesh(centre, 2.5f, level, excluded)
      withClue(s"level $level, excluded $excluded: ") {
        mesh.vertexStride shouldBe reference.vertexStride
        mesh.indices shouldBe reference.indices
        mesh.vertices shouldBe reference.vertices
      }

  "SpongeByVolume.triangleCount" should "match the generated mesh" in:
    for level <- 0 to 3 do
      SpongeByVolume.triangleCount(level) shouldBe
        SpongeByVolume(Vector.Zero[3], 1f, level.toFloat).toTriangleMesh.numTriangles.toLong

  private def legacyFaceMesh(f: Face): menger.common.TriangleMeshData =
    val half = f.scale / 2
    val n = f.normal.toFloatArray
    val offsets: Seq[(Float, Float, Float)] = f.normal match
      case Direction.X | Direction.negX =>
        Seq((0f, -half, -half), (0f, -half, half), (0f, half, half), (0f, half, -half))
      case Direction.Y | Direction.negY =>
        Seq((-half, 0f, -half), (half, 0f, -half), (half, 0f, half), (-half, 0f, half))
      case _ =>
        Seq((-half, -half, 0f), (half, -half, 0f), (half, half, 0f), (-half, half, 0f))
    val uvs = Seq((0f, 0f), (1f, 0f), (1f, 1f), (0f, 1f))
    val verts = offsets.zip(uvs).flatMap { case ((dx, dy, dz), (u, v)) =>
      Seq(f.xCen + dx, f.yCen + dy, f.zCen + dz, n(0), n(1), n(2), u, v)
    }.toArray
    menger.common.TriangleMeshData(verts, Array(0, 1, 2, 0, 2, 3), 8)

  private def legacyMesh(
    centre: Vector[3], scale: Float, level: Int, excluded: Set[Direction]
  ): menger.common.TriangleMeshData =
    if level <= 0 then
      val half = scale / 2
      val faces = Seq(
        Face(centre.x + half, centre.y, centre.z, scale, Direction.X),
        Face(centre.x - half, centre.y, centre.z, scale, Direction.negX),
        Face(centre.x, centre.y + half, centre.z, scale, Direction.Y),
        Face(centre.x, centre.y - half, centre.z, scale, Direction.negY),
        Face(centre.x, centre.y, centre.z + half, scale, Direction.Z),
        Face(centre.x, centre.y, centre.z - half, scale, Direction.negZ)
      ).filterNot(f => excluded.contains(f.normal))
      menger.common.TriangleMeshData.merge(faces.map(legacyFaceMesh))
    else
      val shift = scale / 3f
      val filled: Set[(Int, Int, Int)] =
        (for xx <- -1 to 1; yy <- -1 to 1; zz <- -1 to 1
             if math.abs(xx) + math.abs(yy) + math.abs(zz) > 1
          yield (xx, yy, zz)).toSet
      val subMeshes = for (xx, yy, zz) <- filled.toSeq yield
        val neighbour = Direction.values.filter(d => filled.contains((xx + d.x, yy + d.y, zz + d.z))).toSet
        val inherited = excluded.filter(d => d.x * xx + d.y * yy + d.z * zz == 1)
        legacyMesh(
          Vector[3](centre.x + xx * shift, centre.y + yy * shift, centre.z + zz * shift),
          scale / 3f, level - 1, neighbour ++ inherited
        )
      menger.common.TriangleMeshData.merge(subMeshes.filter(_.numVertices > 0))

  "SpongeByVolume.toString" should "include level" in:
    SpongeByVolume(Vector.Zero[3], 1.0f, level = 1).toString should include ("level=1")
