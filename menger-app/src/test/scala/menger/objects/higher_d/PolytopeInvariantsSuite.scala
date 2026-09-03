package menger.objects.higher_d

import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** A `Mesh4D` whose vertices/faces/cells are fully caller-controlled -- lets tests construct
  * specific, synthetic violations of `PolytopeInvariants`' checks without needing a concrete
  * polytope class to happen to be degenerate in exactly the right way. Wraps a real
  * `Pentachoron`'s geometry by default so any field left unspecified is still a valid,
  * self-consistent 4D simplex. */
final class FakeMesh4D(
  verts: Seq[Vector[4]] = Pentachoron().vertices,
  fcs: Seq[Face4D[3]] = Pentachoron().faces,
  cls: Seq[Seq[Vector[4]]] = Pentachoron().cells
) extends Mesh4D:
  type V = 3
  def vertices: Seq[Vector[4]] = verts
  lazy val faces: Seq[Face4D[V]] = fcs
  override def cells: Seq[Cell4D] = cls

class PolytopeInvariantsSuite extends AnyFlatSpec with Matchers:

  "PolytopeInvariants.check" should "return no findings for a genuinely sound polytope" in:
    PolytopeInvariants.check(Pentachoron()) shouldBe empty

  it should "flag a NaN vertex coordinate" in:
    val verts = Pentachoron().vertices
    val corrupted = verts.updated(0, Vector[4](Float.NaN, verts(0)(1), verts(0)(2), verts(0)(3)))
    val findings = PolytopeInvariants.check(FakeMesh4D(verts = corrupted))
    findings.map(_.invariant) should contain("finite-vertex-coordinates")

  it should "flag an infinite vertex coordinate" in:
    val verts = Pentachoron().vertices
    val corrupted = verts.updated(0, Vector[4](Float.PositiveInfinity, verts(0)(1), verts(0)(2), verts(0)(3)))
    val findings = PolytopeInvariants.check(FakeMesh4D(verts = corrupted))
    findings.map(_.invariant) should contain("finite-vertex-coordinates")

  it should "flag duplicate vertices" in:
    val verts = Pentachoron().vertices
    val withDuplicate = verts :+ verts.head
    val findings = PolytopeInvariants.check(FakeMesh4D(verts = withDuplicate))
    findings.map(_.invariant) should contain("no-duplicate-vertices")

  it should "flag vertices not all on a common sphere" in:
    val verts = Pentachoron().vertices
    val offSphere = verts.updated(0, verts(0) * 5f)
    val findings = PolytopeInvariants.check(FakeMesh4D(verts = offSphere))
    findings.map(_.invariant) should contain("common-sphere")

  it should "flag a mesh not centered at the origin" in:
    val shift = Vector[4](10f, 0f, 0f, 0f)
    val shifted = Pentachoron().vertices.map(_ + shift)
    val findings = PolytopeInvariants.check(FakeMesh4D(verts = shifted))
    findings.map(_.invariant) should contain("centered-at-origin")

  it should "flag a zero-area (degenerate) face" in:
    val v = Pentachoron().vertices.head
    val degenerateFace = Face4D[3](IndexedSeq(v, v, v))
    val findings = PolytopeInvariants.check(FakeMesh4D(fcs = Pentachoron().faces :+ degenerateFace))
    findings.map(_.invariant) should contain("no-degenerate-faces")

  it should "flag an Euler-Poincare violation when cell count is wrong" in:
    val brokenCells = Pentachoron().cells.drop(1) // 5 -> 4 cells, V-E+F-C != 0
    val findings = PolytopeInvariants.check(FakeMesh4D(cls = brokenCells))
    findings.map(_.invariant) should contain("euler-poincare")

  it should "not check Euler-Poincare when the mesh reports no cell structure" in:
    val findings = PolytopeInvariants.check(FakeMesh4D(cls = Seq.empty))
    findings.map(_.invariant) should not contain "euler-poincare"

  it should "report multiple independent findings when several invariants are violated" in:
    val verts = Pentachoron().vertices
    val corrupted = verts.updated(0, Vector[4](Float.NaN, 0f, 0f, 0f))
    val corruptedWithDuplicate = corrupted :+ corrupted(1) // duplicate an untouched vertex
    val findings = PolytopeInvariants.check(FakeMesh4D(verts = corruptedWithDuplicate))
    findings.map(_.invariant) should contain allOf ("finite-vertex-coordinates", "no-duplicate-vertices")
