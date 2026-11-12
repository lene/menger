package menger.objects.higher_d

import com.typesafe.scalalogging.LazyLogging
import menger.Slow
import menger.common.TriangleMeshData
import menger.common.Vector
import menger.engines.scene.ProjectionConversion
import menger.spec.ProjectionSpec
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** The nD Menger sponge geometry (menger#166): its faces against the 4D sponges, the E2
  * prototype's counts and the symmetries of the cube. */
class NdSpongeSuite extends AnyFlatSpec with Matchers with LazyLogging:

  import NdSpongeVariant.{Surface, Volume}

  private type FaceKey = Seq[Seq[Int]]

  private val KeyScale = 1e5f
  private val Dimensions = 4 to 8

  private def faceKey(vertices: Seq[Seq[Float]]): FaceKey =
    vertices.map(_.map(x => math.round(x * KeyScale))).sortBy(_.mkString(","))

  private def faceKeys(mesh: IndexedMeshN): Seq[FaceKey] =
    (0 until mesh.faceCount).map(face => faceKey(mesh.faceVertices(face)))

  private def faceSet(mesh: IndexedMeshN): Set[FaceKey] =
    val keys = faceKeys(mesh)
    withClue(s"duplicate faces: ") { keys.distinct.size shouldBe keys.size }
    keys.toSet

  private def faceSetOf(faces: Seq[Face4D[?]]): Set[FaceKey] =
    faces.map(face => faceKey(face.asSeq.map(vertex => (0 until 4).map(vertex(_))))).toSet

  private def cubeFaceSet(n: Int): Set[FaceKey] =
    val d = Dimension.of(n)
    given ValueOf[d.N] = d.valueOf
    NCube[d.N](1f).faces.map(face =>
      faceKey(face.vertices.map(_.toIndexedSeq))
    ).toSet

  // The counts of the E2 prototype (_bmad-output/forge/dimension-ladder-scene/e2-prototype):
  // counts.txt, and where that has none the prototype itself, run read-only from its directory:
  //   uv run --no-project --with numpy python -c \
  //     "import ndsponge as m; print(len(m.volume_faces('B', 5, 2)))"
  // The rule "A" is k = 1, "B" is k = n - 2.
  private case class Row(variant: NdSpongeVariant, n: Int, level: Int, k: Int, faces: Int)

  private val prototypeRows: Seq[Row] =
    val volumeA = Seq((5, 5120), (6, 30720), (7, 172032), (8, 917504)).map((n, count) =>
      Row(Volume, n, 1, 1, count)
    )
    // volume_faces('B', 8, 1) -> 1025024; counts.txt only has "~1.2e+07 raw" for it
    val volumeB = Seq((5, 5440), (6, 33600), (7, 190848), (8, 1025024)).map((n, count) =>
      Row(Volume, n, 1, n - 2, count)
    )
    val surface = Seq((5, 1600), (6, 5760), (7, 18816), (8, 57344)).flatMap((n, count) =>
      Seq(Row(Surface, n, 1, 1, count), Row(Surface, n, 1, n - 2, count))
    )
    val levelTwo = Seq(
      Row(Volume, 5, 2, 1, 450560),
      Row(Volume, 5, 2, 3, 539200), // volume_faces('B', 5, 2) -> 539200
      Row(Surface, 5, 2, 1, 28160),
      Row(Surface, 5, 2, 3, 32000),
      Row(Surface, 6, 2, 1, 122880),
      Row(Surface, 6, 2, 4, 138240)
    )
    val fourD = Seq(
      Row(Volume, 4, 2, 2, 30336), Row(Surface, 4, 2, 2, 6144),
      Row(Volume, 4, 1, 2, 768), Row(Surface, 4, 1, 2, 384)
    )
    volumeA ++ volumeB ++ surface ++ levelTwo ++ fourD

  "volume(4, L, 1)" should "have the faces of TesseractSponge(L), 24 / 768 / 29,184 of them" in:
    Seq(0 -> 24, 1 -> 768, 2 -> 29184).foreach { (level, count) =>
      val mesh = NdSponge.volume(4, level, 1)
      withClue(s"level $level: ") {
        mesh.faceCount shouldBe count
        faceSet(mesh) shouldBe faceSetOf(TesseractSponge(level).faces)
      }
    }

  "surface(4, L, 1)" should "have the faces of TesseractSponge2(L), 24 / 384 / 5,376 of them" in:
    Seq(0 -> 24, 1 -> 384, 2 -> 5376).foreach { (level, count) =>
      val mesh = NdSponge.surface(4, level, 1)
      withClue(s"level $level: ") {
        mesh.faceCount shouldBe count
        faceSet(mesh) shouldBe faceSetOf(TesseractSponge2(level).faces)
      }
    }

  "Level 0" should "be the n-cube, for both variants" in:
    Dimensions.foreach { n =>
      Seq(Volume, Surface).foreach { variant =>
        withClue(s"$variant, n = $n: ") {
          val mesh = NdSponge(variant, n, 0)
          mesh.faceCount shouldBe NCube.faceCount(n)
          mesh.vertexCount shouldBe NCube.vertexCount(n)
          faceSet(mesh) shouldBe cubeFaceSet(n)
        }
      }
    }

  "The sponges" should "have the face counts of the prototype" in:
    prototypeRows.foreach { case Row(variant, n, level, k, faces) =>
      withClue(s"$variant n = $n, level $level, k = $k: ") {
        NdSponge(variant, n, level, k).faceCount shouldBe faces
      }
    }

  they should "be bounded above by upperBoundFaces, exactly so for the 4D surface" in:
    prototypeRows.foreach { case Row(variant, n, level, k, faces) =>
      withClue(s"$variant n = $n, level $level, k = $k: ") {
        NdSponge.upperBoundFaces(variant, n, level, k) should be >= faces.toLong
      }
    }
    NdSponge.upperBoundFaces(Volume, 4, 1, 1) shouldBe 1152L

  they should "keep the 8D bound from overflowing" in:
    NdSponge.upperBoundFaces(Volume, 8, 12, 1) shouldBe Long.MaxValue
    NdSponge.upperBoundFaces(Surface, 8, 0, 1) shouldBe NCube.faceCount(8).toLong

  they should "give the bound of a huge level at once, saturated" in:
    val (bounds, millis) = timed(
      Seq(Volume, Surface).map(NdSponge.upperBoundFaces(_, 8, Int.MaxValue, 1))
    )
    bounds shouldBe Seq(Long.MaxValue, Long.MaxValue)
    millis should be < 1000L

  they should "keep 48, 112 and 1,280 sub-cubes per step with k = 1 in 4D, 5D and 8D" in:
    Seq(4 -> 48, 5 -> 112, 8 -> 1280).foreach { (n, kept) =>
      NdSponge.keptOffsets(n, 1) should have size kept
      NdSponge.keptCells(n, 1, 1).length shouldBe kept * n
    }
    NdSponge.keptOffsets(4, 2) should have size 72
    NdSponge.keptCells(4, 2, 1).length shouldBe 48 * 48 * 4

  they should "not move when the axes are permuted or a sign is flipped (5D, level 1)" in:
    val n = 5
    for variant <- Seq(Volume, Surface); k <- Seq(1, n - 2) do
      val keys = faceSet(NdSponge(variant, n, 1, k))
      def transformed(move: Seq[Int] => Seq[Int]): Set[FaceKey] =
        keys.map(_.map(move).sortBy(_.mkString(",")))
      withClue(s"$variant, k = $k: ") {
        (0 until n).permutations.foreach { permutation =>
          transformed(vertex => permutation.map(vertex)) shouldBe keys
        }
        (0 until n).foreach { axis =>
          transformed(vertex => vertex.updated(axis, -vertex(axis))) shouldBe keys
        }
      }

  they should "lie in the unit n-cube" in:
    Seq(Volume, Surface).foreach { variant =>
      NdSponge(variant, 5, 2, 1).vertices.foreach(math.abs(_) should be <= 0.5f)
    }

  they should "reach +-0.5 on every axis at level 1" in:
    Dimensions.foreach { n =>
      Seq(Volume, Surface).foreach { variant =>
        val points = NdSponge(variant, n, 1, 1).vertices.grouped(n).toSeq
        (0 until n).foreach { axis =>
          withClue(s"$variant, n = $n, axis $axis: ") {
            points.map(_(axis)).max shouldBe 0.5f
            points.map(_(axis)).min shouldBe -0.5f
          }
        }
      }
    }

  they should "stay below their upper bound for every k (exhaustive for small n and level)" in:
    val cases = for
      (n, levels) <- Seq(4 -> Seq(1, 2, 3), 5 -> Seq(1, 2), 6 -> Seq(1, 2))
      level <- levels
      variant <- Seq(Volume, Surface)
      k <- 1 until n
    yield (variant, n, level, k)
    cases.foreach { (variant, n, level, k) =>
      withClue(s"$variant n = $n, level $level, k = $k: ") {
        NdSponge(variant, n, level, k).faceCount.toLong should
          be <= NdSponge.upperBoundFaces(variant, n, level, k)
      }
    }

  they should "have a surface bound of 8 + 8(n-2) children per face and step" in:
    NdSponge.upperBoundFaces(Surface, 4, 1, 1) shouldBe 24L * 24
    NdSponge.upperBoundFaces(Surface, 8, 2, 1) shouldBe 1792L * 56 * 56

  "maxCentreCoordinates outside 1..n-1" should "be rejected, naming the range" in:
    Seq(0, 5).foreach { k =>
      val volume = the[IllegalArgumentException] thrownBy NdSponge.volume(5, 1, k)
      volume.getMessage should include("1..4")
      val surface = the[IllegalArgumentException] thrownBy NdSponge.surface(5, 1, k)
      surface.getMessage should include("1..4")
      an[IllegalArgumentException] should be thrownBy NdSponge.keptOffsets(5, k)
      an[IllegalArgumentException] should be thrownBy
        NdSponge.upperBoundFaces(Volume, 5, 1, k)
    }

  "A dimension outside 4..8 or a negative level" should "be rejected" in:
    Seq(3, 9).foreach { n =>
      an[IllegalArgumentException] should be thrownBy NdSponge.volume(n, 1, 1)
      an[IllegalArgumentException] should be thrownBy NdSponge.surface(n, 1, 1)
    }
    an[IllegalArgumentException] should be thrownBy NdSponge.volume(5, -1, 1)
    an[IllegalArgumentException] should be thrownBy NdSponge.surface(5, -1, 1)

  "A level whose face key does not fit in a Long" should "be rejected, naming N and L" in:
    Seq[() => IndexedMeshN](() => NdSponge.volume(8, 4, 1), () => NdSponge.surface(8, 4, 1))
      .foreach { build =>
        val message = (the[IllegalArgumentException] thrownBy build()).getMessage
        message should (include("N=8") and include("L=4") and include("too wide"))
      }

  "A sponge too large for an array" should "be rejected before anything is built" in:
    // 1280^3 kept cells of 8D; 7D level 4 passes the key width but has 3.6e9 candidate faces
    Seq[() => Any](
      () => NdSponge.volume(8, 3, 1), () => NdSponge.keptCells(8, 3, 1),
      () => NdSponge.surface(7, 4, 1), () => NdSponge.surface(6, 5, 1)
    ).foreach { build =>
      val (failure, millis) = timed(the[IllegalArgumentException] thrownBy build())
      failure.getMessage should include("too large to build")
      millis should be < 1000L
    }

  "A huge level" should "fail fast with the key-width message" in:
    Seq[() => Any](
      () => NdSponge.volume(5, Int.MaxValue, 1), () => NdSponge.surface(5, Int.MaxValue, 1),
      () => NdSponge.keptCells(5, Int.MaxValue, 1)
    ).foreach { build =>
      val (failure, millis) = timed(the[IllegalArgumentException] thrownBy build())
      failure.getMessage should (include("too wide") and include("L=2147483647"))
      millis should be < 1000L
    }

  "The face key" should "round-trip at the widest accepted keys" in:
    // n = 8 at level 3 uses 54 of the 63 key bits, n = 4 at level 8 uses 62
    Seq((8, 3, 6, 7), (4, 8, 2, 3)).foreach { (n, level, i, j) =>
      val keys = NdSponge.keysFor(n, level, 1)
      val extent = keys.extent
      val centres = Seq(
        Array.fill(n)(extent), Array.fill(n)(-extent), Array.fill(n)(0),
        Array.tabulate(n)(axis => if axis % 2 == 0 then extent else -extent)
      )
      centres.foreach { centre =>
        withClue(s"n = $n, level $level, centre ${centre.toSeq}: ") {
          val key = keys.encode(i, j, centre)
          key should be >= 0L
          keys.firstAxis(key) shouldBe i
          keys.secondAxis(key) shouldBe j
          keys.centre(key).toSeq shouldBe centre.toSeq
        }
      }
      val corner = keys.corner(keys.encode(i, j, Array.fill(n)(0)), 1, -1)
      (0 until n).map(keys.coordinate(corner, _)) shouldBe
        (0 until n).map(axis => if axis == i then 1 else if axis == j then -1 else 0)
    }

  private def defaultView(n: Int): LiftedProjection =
    ProjectionConversion.liftedProjection(ProjectionSpec(n))

  private def faceChunks(mesh: TriangleMeshData): List[List[Float]] =
    mesh.vertices.grouped(32).map(_.toList).toList.sortBy(_.toString)

  "Level 0 through the default view" should "give the triangles of NCube.projected" in:
    Dimensions.foreach { n =>
      Seq(Volume, Surface).foreach { variant =>
        val size = 1.5f
        val center = Vector[3](0.25f, -1f, 2f)
        val view = defaultView(n)
        val mesh = NdSponge(variant, n, 0).toTriangleMesh(
          size, IndexedSeq.fill(n)(1f), view.project, center
        )
        val expected = NCube.projected(n, size, view, center).toTriangleMesh
        withClue(s"$variant, n = $n: ") {
          mesh.numTriangles shouldBe expected.numTriangles
          faceChunks(mesh) shouldBe faceChunks(expected)
          mesh.vertexStride shouldBe expected.vertexStride
        }
      }
    }

  "A sponge with the eye inside it" should "drop faces with a clipped vertex, emit the rest" in:
    // the eye is at w = 0.3, inside the sponge, which reaches w = 0.5
    val view = LiftedProjection(4, Nil, Seq(Projection(0.3f, 0.2f)))
    val mesh = NdSponge.volume(4, 1, 1)
    val visible = (0 until mesh.faceCount).count(face =>
      mesh.faceVertices(face).forall(vertex => view.project(vertex).isDefined)
    )
    visible should (be > 0 and be < mesh.faceCount)
    val triangles = mesh.toTriangleMesh(1f, IndexedSeq.fill(4)(1f), view.project, Vector.Zero[3])
    triangles.numTriangles shouldBe 2 * visible

  private def timed[A](work: => A): (A, Long) =
    val start = System.nanoTime()
    val result = work
    (result, (System.nanoTime() - start) / 1000000)

  "Building a large sponge" should "give the known face counts and log the times" taggedAs Slow in:
    // counts.txt and the E2 card: 5D and 6D level 2 with rule A, 8D level 2 with rule S
    Seq((Volume, 5, 2, 450560), (Volume, 6, 2, 5898240), (Surface, 8, 2, 1662976)).foreach {
      (variant, n, level, faces) =>
        val (mesh, buildMillis) = timed(NdSponge(variant, n, level, 1))
        mesh.faceCount shouldBe faces
        val view = defaultView(n)
        val (triangles, projectMillis) = timed(
          mesh.toTriangleMesh(1f, IndexedSeq.fill(n)(1f), view.project, Vector.Zero[3])
        )
        logger.info(
          s"$variant n = $n level $level: ${mesh.faceCount} faces, ${mesh.vertexCount} " +
            s"vertices, built in $buildMillis ms; toTriangleMesh $projectMillis ms " +
            s"(${triangles.numTriangles} triangles)"
        )
    }
