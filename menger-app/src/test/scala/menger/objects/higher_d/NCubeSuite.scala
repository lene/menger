package menger.objects.higher_d

import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** The n-cube mesh (menger#163): counts, shape and order for every dimension 4..8, and the
  * 4-cube against the hand-written [[Tesseract]]. */
class NCubeSuite extends AnyFlatSpec with Matchers:

  private val Dimensions = 4 to 8
  private val Size = 1.5f
  private val Tolerance = 1e-5f

  private def cube(dimension: Int, size: Float = Size): NCube[?] =
    val d = Dimension.of(dimension)
    given ValueOf[d.N] = d.valueOf
    NCube[d.N](size)

  private def coordinates(vertex: Vector[?]): IndexedSeq[Float] = vertex.toIndexedSeq

  private def choose(n: Int, k: Int): Long =
    (1 to k).foldLeft(1L)((acc, i) => acc * (n - k + i) / i)

  private def distance(a: Vector[?], b: Vector[?]): Float =
    math.sqrt(coordinates(a).zip(coordinates(b)).map((x, y) => (x - y) * (x - y)).sum).toFloat

  private def differingAxes(a: Vector[?], b: Vector[?]): Seq[Int] =
    coordinates(a).zip(coordinates(b)).zipWithIndex.collect {
      case ((x, y), axis) if x != y => axis
    }

  "An n-cube" should "have 2^N vertices, N * 2^(N-1) edges and C(N,2) * 2^(N-2) faces" in:
    Dimensions.foreach { n =>
      val c = cube(n)
      c.vertices.size shouldBe (1 << n)
      c.orderedEdges.size shouldBe n * (1 << (n - 1))
      c.faces.size shouldBe (choose(n, 2) * (1L << (n - 2))).toInt
      NCube.vertexCount(n) shouldBe c.vertices.size
      NCube.edgeCount(n) shouldBe c.orderedEdges.size
      NCube.faceCount(n) shouldBe c.faces.size
    }

  it should "have the counts of the penteract and the 8-cube from the spec" in:
    (cube(5).vertices.size, cube(5).orderedEdges.size, cube(5).faces.size) shouldBe ((32, 80, 80))
    (cube(8).orderedEdges.size, cube(8).faces.size) shouldBe ((1024, 1792))

  it should "have only distinct vertices at +-size/2 on every axis" in:
    Dimensions.foreach { n =>
      val c = cube(n)
      c.vertices.map(coordinates).distinct.size shouldBe c.vertices.size
      c.vertices.foreach(v => coordinates(v).foreach(x => math.abs(x) shouldBe Size / 2))
    }

  it should "give every vertex exactly N edges" in:
    Dimensions.foreach { n =>
      val c = cube(n)
      val degrees = c.orderedEdges.flatMap((from, to) => Seq(coordinates(from), coordinates(to)))
        .groupBy(identity).view.mapValues(_.size)
      degrees.size shouldBe c.vertices.size
      degrees.values.toSet shouldBe Set(n)
    }

  it should "have edges of length size along one axis" in:
    Dimensions.foreach { n =>
      cube(n).orderedEdges.foreach { (from, to) =>
        distance(from, to) shouldBe Size +- Tolerance
        differingAxes(from, to).size shouldBe 1
      }
    }

  it should "list each edge once, the lexicographically smaller endpoint first" in:
    Dimensions.foreach { n =>
      val c = cube(n)
      val keys = c.orderedEdges.map((from, to) => (coordinates(from), coordinates(to)))
      keys.distinct.size shouldBe keys.size
      keys.foreach { (from, to) =>
        from.zip(to).find(_ != _).forall(_ < _) shouldBe true
      }
    }

  it should "keep its edges in the same order every time" in:
    Dimensions.foreach { n =>
      val first = cube(n).orderedEdges.map((a, b) => (coordinates(a), coordinates(b)))
      val second = cube(n).orderedEdges.map((a, b) => (coordinates(a), coordinates(b)))
      first shouldBe second
    }

  it should "have the same edges unordered as the edges of its faces" in:
    Dimensions.foreach { n =>
      val c = cube(n)
      c.edges.size shouldBe c.orderedEdges.size
      c.edges.map((a, b) => (coordinates(a), coordinates(b))) shouldBe
        c.orderedEdges.map((a, b) => (coordinates(a), coordinates(b))).toSet
    }

  it should "have square faces, each in one pair of axes" in:
    Dimensions.foreach { n =>
      cube(n).faces.foreach { face =>
        face.vertsPerFace shouldBe 4
        val corners = face.vertices
        val axes = corners.tail.flatMap(differingAxes(corners.head, _)).distinct
        axes.size shouldBe 2
        val sides = corners.indices.map(i => distance(corners(i), corners((i + 1) % 4)))
        sides.foreach(_ shouldBe Size +- Tolerance)
        distance(corners(0), corners(2)) shouldBe (Size * math.sqrt(2.0).toFloat) +- Tolerance
        distance(corners(1), corners(3)) shouldBe (Size * math.sqrt(2.0).toFloat) +- Tolerance
      }
    }

  it should "have one face for each pair of axes and choice of signs on the other axes" in:
    Dimensions.foreach { n =>
      val c = cube(n)
      val keys = c.faces.map { face =>
        val corners = face.vertices
        val axes = corners.tail.flatMap(differingAxes(corners.head, _)).distinct.sorted
        val fixed = (0 until n).diff(axes).map(corners.head.toIndexedSeq)
        (axes, fixed)
      }
      keys.distinct.size shouldBe c.faces.size
    }

  it should "have the Euler characteristic of the boundary of an N-ball" in:
    // The boundary is an (N-1)-sphere: the alternating sum of its k-face counts C(N,k) 2^(N-k),
    // k < N, is 1 + (-1)^(N-1): 2 for odd N, 0 for even. The k = 0, 1, 2 counts are the mesh's.
    Dimensions.foreach { n =>
      val c = cube(n)
      val meshCounts = Seq(c.vertices.size.toLong, c.orderedEdges.size.toLong, c.faces.size.toLong)
      val formulaCounts = (0 until n).map(k => choose(n, k) * (1L << (n - k)))
      formulaCounts.take(3) shouldBe meshCounts
      val euler = formulaCounts.zipWithIndex.map((count, k) =>
        if k % 2 == 0 then count else -count
      ).sum
      euler shouldBe (if n % 2 == 0 then 0L else 2L)
    }

  "The 4-cube" should "have the vertices of the Tesseract, in its order" in:
    val tesseract = Tesseract(Size)
    cube(4).vertices.map(coordinates) shouldBe tesseract.vertices.map(coordinates)

  it should "have the faces of the Tesseract, as sets of corners" in:
    def cornerSets(faces: Seq[Polygon[?, ?]]): Set[Set[IndexedSeq[Float]]] =
      faces.map(_.vertices.map(coordinates).toSet).toSet
    cornerSets(cube(4).faces) shouldBe cornerSets(Tesseract(Size).faces)

  it should "have the edges of the Tesseract" in:
    val tesseract = Tesseract(Size)
    cube(4).orderedEdges.map((a, b) => (coordinates(a), coordinates(b))).toSet shouldBe
      tesseract.edges.map((a, b) => (coordinates(a), coordinates(b)))

  "NCube.projected" should "triangulate every face into two triangles" in:
    Dimensions.foreach { n =>
      val view = LiftedProjection(n, Nil, Seq.fill(n - 3)(Projection(6f, 1.5f)))
      NCube.projected(n, Size, view).toTriangleMesh.numTriangles shouldBe 2 * NCube.faceCount(n)
    }

  it should "put the cube's vertices where the view puts them" in:
    val view = LiftedProjection(6, Nil, Seq.fill(3)(Projection(6f, 1.5f)))
    val mesh = NCube.projected(6, Size, view).toTriangleMesh
    val positions = mesh.vertices.grouped(mesh.vertexStride).map(_.take(3).toSeq).toSet
    val expected = cube(6).vertices.map(v => view(coordinates(v))).map(p => Seq(p(0), p(1), p(2)))
    positions shouldBe expected.toSet

  "NCube.projectedEdges" should "give one entry per edge, in the order of the edges" in:
    Dimensions.foreach { n =>
      val view = LiftedProjection(n, Nil, Seq.fill(n - 3)(Projection(6f, 1.5f)))
      val endpoints = NCube.projectedEdges(n, Size, view)
      endpoints.size shouldBe NCube.edgeCount(n)
      endpoints.zip(cube(n).orderedEdges).foreach { case (Some((p0, p1)), (from, to)) =>
        p0 shouldBe view(coordinates(from))
        p1 shouldBe view(coordinates(to))
      case (None, _) => fail("no edge is clipped at eye 6")
      }
    }

  "A cube lifted into a higher dimension" should "be turned by planes using the extra axes" in:
    val stages = Seq.fill(4)(Projection(6f, 1.5f))
    val plain = LiftedProjection(7, Nil, stages)
    val turned = LiftedProjection(
      7, Seq((Plane(4, 5, 7), 40f), (Plane(5, 6, 7), 30f), (Plane(0, 6, 7), 25f)), stages
    )
    val vertex = coordinates(cube(5).vertices.last)
    plain(vertex) should not be turned(vertex)
    NCube.projected(5, Size, turned).toTriangleMesh.vertices should not be
      NCube.projected(5, Size, plain).toTriangleMesh.vertices

  it should "be shrunk by the stages of the extra axes when it is not turned" in:
    // An unturned 5-cube in 7D has t = u = 0: each of the two extra stages scales every
    // coordinate by (eye - screen) / eye, then the 5D chain proceeds as it would without them.
    val eye = 6f
    val screen = 1.5f
    val extraStage = (eye - screen) / eye
    val lifted = LiftedProjection(7, Nil, Seq.fill(4)(Projection(eye, screen)))
    val flat = LiftedProjection(5, Nil, Seq.fill(2)(Projection(eye, screen)))
    cube(5).vertices.map(coordinates).foreach { vertex =>
      val expected = flat(vertex.map(_ * extraStage * extraStage))
      val actual = lifted(vertex)
      (0 until 3).foreach(axis => actual(axis) shouldBe expected(axis) +- Tolerance)
    }

  it should "be refused when the view has fewer dimensions than the cube" in:
    val view = LiftedProjection(5, Nil, Seq.fill(2)(Projection(6f, 1.5f)))
    val refusal = the[IllegalArgumentException] thrownBy
      NCube.projected(6, Size, view).toTriangleMesh
    refusal.getMessage should (include("6") and include("5"))

  "A stage with its eye inside the cube" should "drop the faces and edges past the eye" in:
    // Size 2: the coordinates are +-1; the first stage drops axis v with eye 0.6, so every
    // vertex with v = +1 is at or behind the eye.
    val view = LiftedProjection(5, Nil, Seq(Projection(0.6f, 0.3f), Projection(6f, 1.5f)))
    val faces = NCube.projected(5, 2f, view).toTriangleMesh
    // Faces left: a pair of the four axes x y z w (6) and a sign on each of the other three,
    // v being one of them and fixed at -1 (4 of 8 combinations).
    faces.numTriangles shouldBe 2 * 24
    val edges = NCube.projectedEdges(5, 2f, view)
    edges.flatten.size shouldBe 32
    edges.count(_.isEmpty) shouldBe NCube.edgeCount(5) - 32
    faces.vertices.forall(_.isFinite) shouldBe true
