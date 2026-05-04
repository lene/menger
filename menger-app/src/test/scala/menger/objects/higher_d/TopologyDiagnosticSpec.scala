package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Diagnostic — not a regression test.
  *
  * Runs MeshTopology checkers on TesseractSponge and TesseractSponge2 at
  * levels 0–2, at both the 4D (pre-projection) and 3D (post-projection)
  * stages. Output goes to stdout and feeds the Stage 2 section of the
  * investigation document.
  *
  * Tests are marked `ignore` so they don't run in normal CI; reactivate by
  * changing `ignore` to `it` when gathering evidence.
  */
class TopologyDiagnosticSpec extends AnyFlatSpec with Matchers:

  private def report(label: String, r: MeshTopology.TopologyReport): Unit =
    val manifold = if r.isManifold then "MANIFOLD" else "NON-MANIFOLD"
    println(s"  $label → $manifold | faces=${r.faceCount} | hist=${r.edgeUseHistogram} | boundaryFaces=${r.boundaryFaces.size}")
    if r.boundaryFaces.nonEmpty then
      println(s"    first boundary face indices: ${r.boundaryFaces.take(5)}")

  "TopologyDiagnostic" should "print TesseractSponge2 topology at levels 0-2 (4D and 3D)" ignore:
    println("\n=== TesseractSponge2 ===")
    for level <- Seq(0, 1, 2) do
      println(s"--- Level $level ---")
      val faces4D = TesseractSponge2(level).faces
      val mesh3D  = TesseractSponge2Mesh(level = level).toTriangleMesh
      report(s"4D (${faces4D.size} faces)", MeshTopology.checkFace4D(faces4D))
      report(s"3D (${mesh3D.numTriangles} triangles)", MeshTopology.checkTriangleMesh(mesh3D))

  it should "run sub-cube and vertex-position checks on level-2 boundary faces" ignore:
    val faces  = TesseractSponge2(2).faces
    val report = MeshTopology.checkFace4D(faces)
    println(s"\n=== Level-2 boundary diagnosis: ${report.boundaryEdgeCount} boundary edges ===")

    // 1. Sub-cube test: all level-2 face corners must be inside at least one level-1 sub-cube.
    val subCubes = level1SubCubes(size = 1f)
    val outsideFaces = faces.zipWithIndex.filter { case (f, _) => !faceInsideSponge(f, subCubes) }
    println(s"  Sub-cube test: ${outsideFaces.size} faces with corners outside level-1 sponge")
    outsideFaces.take(2).foreach { case (f, i) =>
      println(s"    face[$i]: ${f.asSeq.map(fmt).mkString(", ")}")
    }

    // 2. For boundary faces: does any other face in the mesh share the same 4 vertex positions?
    //    If yes → wrong vertex ordering ("flipped"). If no → structural gap.
    type VSet = Set[(Float,Float,Float,Float)]
    def vset(f: Face4D[4]): VSet = f.asSeq.map(v => (v(0), v(1), v(2), v(3))).toSet
    val allVSets = faces.map(vset)
    val bfSample = report.boundaryFaces.take(5).map(faces(_))
    println(s"\n  Vertex-position check on first ${bfSample.size} boundary faces:")
    for (bf, idx) <- bfSample.zipWithIndex do
      val vs    = vset(bf)
      val twins = allVSets.count(_ == vs) - 1  // subtract self
      println(s"    BoundaryFace[$idx]: ${bf.asSeq.map(fmt).mkString(", ")}")
      println(s"      other faces with same vertex positions: $twins")

  private def level1SubCubes(size: Float): Seq[(Float,Float,Float,Float,Float,Float,Float,Float)] =
    val half  = size / 2f
    val third = size / 3f
    for
      i <- 0 to 2
      j <- 0 to 2
      k <- 0 to 2
      l <- 0 to 2
      if Seq(i, j, k, l).count(_ == 1) <= 1
    yield
      ( -half + i*third, -half + (i+1)*third,
        -half + j*third, -half + (j+1)*third,
        -half + k*third, -half + (k+1)*third,
        -half + l*third, -half + (l+1)*third )

  private def isInsideCube(
      v: menger.common.Vector[4],
      c: (Float,Float,Float,Float,Float,Float,Float,Float),
      eps: Float = 1e-4f
  ): Boolean =
    v(0) >= c._1-eps && v(0) <= c._2+eps &&
    v(1) >= c._3-eps && v(1) <= c._4+eps &&
    v(2) >= c._5-eps && v(2) <= c._6+eps &&
    v(3) >= c._7-eps && v(3) <= c._8+eps

  private def faceInsideSponge(
      face: Face4D[4],
      subCubes: Seq[(Float,Float,Float,Float,Float,Float,Float,Float)]
  ): Boolean =
    face.asSeq.forall(v => subCubes.exists(c => isInsideCube(v, c)))

  private def fmt(v: menger.common.Vector[4]): String =
    f"(${v(0)}%.6f, ${v(1)}%.6f, ${v(2)}%.6f, ${v(3)}%.6f)"

  it should "print TesseractSponge topology at levels 0-2 (4D and 3D)" ignore:
    println("\n=== TesseractSponge (sibling) ===")
    for level <- Seq(0, 1, 2) do
      println(s"--- Level $level ---")
      val faces4D = TesseractSponge(level).faces
      val mesh3D  = TesseractSpongeMesh(level = level).toTriangleMesh
      report(s"4D (${faces4D.size} faces)", MeshTopology.checkFace4D(faces4D))
      report(s"3D (${mesh3D.numTriangles} triangles)", MeshTopology.checkTriangleMesh(mesh3D))
