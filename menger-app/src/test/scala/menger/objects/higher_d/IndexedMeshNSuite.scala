package menger.objects.higher_d

import java.util.concurrent.atomic.AtomicInteger

import menger.common.TriangleMeshData
import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** The indexed nD quad mesh (menger#166): face access, hole caps and the projection to
  * triangles. */
class IndexedMeshNSuite extends AnyFlatSpec with Matchers:

  private val Tolerance = 1e-6f

  // two unit squares in the xy plane of 4D, sharing the edge between vertices 1 and 2
  private val twoSquares = IndexedMeshN(
    4,
    Array(
      0f, 0f, 0f, 0f,  1f, 0f, 0f, 0f,  1f, 1f, 0f, 0f,
      0f, 1f, 0f, 0f,  2f, 0f, 0f, 0f,  2f, 1f, 0f, 0f
    ),
    Array(0, 1, 2, 3,  1, 4, 5, 2)
  )

  private def firstThree(coordinates: IndexedSeq[Float]): Option[Vector[3]] =
    Some(Vector[3](coordinates(0), coordinates(1), coordinates(2)))

  private val unscaled = IndexedSeq.fill(4)(1f)
  private val origin = Vector[3](0f, 0f, 0f)

  private def positions(mesh: TriangleMeshData): Set[Seq[Float]] =
    mesh.vertices.grouped(mesh.vertexStride).map(_.take(3).toSeq).toSet

  private def squareArea(face: IndexedSeq[IndexedSeq[Float]]): Double =
    def edge(from: Int, to: Int): Double =
      math.sqrt(face(from).zip(face(to)).map((a, b) => (a - b).toDouble * (a - b)).sum)
    edge(0, 1) * edge(0, 3)

  private def centroid(face: IndexedSeq[IndexedSeq[Float]]): IndexedSeq[Float] =
    face.transpose.map(_.sum / face.size)

  "An IndexedMeshN" should "count its vertices and faces" in:
    (twoSquares.vertexCount, twoSquares.faceCount) shouldBe ((6, 2))

  it should "give a face's vertices as coordinate lists in the face's order" in:
    twoSquares.faceVertices(1) shouldBe IndexedSeq(
      IndexedSeq(1f, 0f, 0f, 0f), IndexedSeq(2f, 0f, 0f, 0f),
      IndexedSeq(2f, 1f, 0f, 0f), IndexedSeq(1f, 1f, 0f, 0f)
    )

  it should "reject vertices that do not fill whole points and indices that do not fill faces" in:
    an[IllegalArgumentException] should be thrownBy IndexedMeshN(4, Array(0f, 0f, 0f), Array.empty)
    an[IllegalArgumentException] should be thrownBy IndexedMeshN(4, Array.empty, Array(0, 1, 2))

  "holeCaps" should "keep the number of faces" in:
    val mesh = NdSponge.volume(4, 1, 1)
    mesh.holeCaps.faceCount shouldBe mesh.faceCount

  it should "make each cap a ninth of its face's area around the face's centroid" in:
    val mesh = NdSponge.surface(5, 1, 1)
    val caps = mesh.holeCaps
    (0 until mesh.faceCount).foreach { face =>
      val whole = mesh.faceVertices(face)
      val cap = caps.faceVertices(face)
      squareArea(cap) shouldBe (squareArea(whole) / 9) +- (Tolerance.toDouble)
      centroid(cap).zip(centroid(whole)).foreach((c, w) => c shouldBe w +- Tolerance)
    }

  it should "give each cap its own four vertices" in:
    val caps = twoSquares.holeCaps
    caps.vertexCount shouldBe 4 * twoSquares.faceCount
    caps.quads.toSeq shouldBe (0 until caps.vertexCount)

  "toTriangleMesh" should "make two triangles of every face" in:
    twoSquares.toTriangleMesh(1f, unscaled, firstThree, origin).numTriangles shouldBe 4

  it should "emit the triangles of the shared triangulation, face by face" in:
    val mesh = twoSquares.toTriangleMesh(1f, unscaled, firstThree, origin)
    val expected = (0 until 2).map { face =>
      val corners = twoSquares.faceVertices(face).flatMap(firstThree)
      FaceTriangles.faceToTriangleMesh(corners)
    }
    mesh.vertices.toSeq shouldBe expected.flatMap(_.vertices)
    mesh.indices.toSeq shouldBe
      (expected.head.indices ++ expected(1).indices.map(_ + 4)).toSeq

  it should "project every vertex once, however many faces share it" in:
    val calls = AtomicInteger(0)
    twoSquares.toTriangleMesh(1f, unscaled, { c => calls.incrementAndGet(); firstThree(c) }, origin)
    calls.get shouldBe twoSquares.vertexCount

  it should "scale each coordinate by the size and then by the scale of its axis" in:
    val mesh = twoSquares.toTriangleMesh(2f, IndexedSeq(1f, 3f, 1f, 1f), firstThree, origin)
    positions(mesh) shouldBe Set(
      Seq(0f, 0f, 0f), Seq(2f, 0f, 0f), Seq(2f, 6f, 0f), Seq(0f, 6f, 0f),
      Seq(4f, 0f, 0f), Seq(4f, 6f, 0f)
    )

  it should "pass the scaled coordinates, all of them, to the projection" in:
    val seen = scala.collection.mutable.Set.empty[IndexedSeq[Float]]
    twoSquares.toTriangleMesh(2f, IndexedSeq(1f, 3f, 1f, 5f), { c => seen += c; None }, origin)
    seen should contain(IndexedSeq(2f, 0f, 0f, 0f))
    seen.foreach(_.size shouldBe 4)

  it should "need one scale per axis" in:
    an[IllegalArgumentException] should be thrownBy
      twoSquares.toTriangleMesh(1f, IndexedSeq(1f, 1f, 1f), firstThree, origin)

  it should "drop a face with a vertex the projection rejects, and keep the rest" in:
    val reject = (c: IndexedSeq[Float]) => if c(0) > 1.5f then None else firstThree(c)
    val mesh = twoSquares.toTriangleMesh(1f, unscaled, reject, origin)
    mesh.numTriangles shouldBe 2
    positions(mesh) shouldBe Set(Seq(0f, 0f, 0f), Seq(1f, 0f, 0f), Seq(1f, 1f, 0f), Seq(0f, 1f, 0f))
    mesh.indices.toSeq shouldBe Seq(0, 1, 2, 0, 2, 3)

  it should "be empty when every face is dropped" in:
    val empty = twoSquares.toTriangleMesh(1f, unscaled, _ => None, origin)
    empty.numTriangles shouldBe 0
    empty.vertexStride shouldBe 8

  it should "move the result to the center" in:
    val mesh = twoSquares.toTriangleMesh(1f, unscaled, firstThree, Vector[3](10f, 20f, 30f))
    positions(mesh) should contain allOf (Seq(10f, 20f, 30f), Seq(12f, 21f, 30f))
    mesh.vertices.grouped(8).map(_.slice(3, 6).toSeq).toSet should not contain Seq(10f, 20f, 30f)
