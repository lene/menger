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

  // The existing degenerate-face case builds three *identical* vertices, so its area is
  // exactly 0f -- it passes identically under the old `<= 0f` comparison and the current
  // epsilon one, meaning nothing pinned the round-1 fix. This is the case that actually
  // distinguishes them: a sliver with a tiny but strictly positive area (review round 2).
  it should "flag a sliver face whose area is tiny but not exactly zero" in:
    // Orthogonal offsets, so the triangle is genuinely non-degenerate: area = 0.5 * 2e-3 *
    // 2e-3 = 2e-6, comfortably representable in float32 yet far below any face the
    // Pentachoron itself contributes.
    val base = Pentachoron().vertices.head
    val sliver = Face4D[3](IndexedSeq(
      base,
      base + Vector[4](2e-3f, 0f, 0f, 0f),
      base + Vector[4](0f, 2e-3f, 0f, 0f)
    ))
    val findings = PolytopeInvariants.check(FakeMesh4D(fcs = Pentachoron().faces :+ sliver))
    withClue(s"sliver area was ${sliver.area}, must be > 0 for this test to mean anything: "):
      sliver.area should be > 0f
    findings.map(_.invariant) should contain("no-degenerate-faces")

  // The other direction of the same fix: an absolute epsilon flagged every face of a
  // legitimately small object as degenerate. Tolerances are relative to the mesh's own scale.
  it should "not flag a legitimately small polytope as degenerate" in:
    PolytopeInvariants.check(Tesseract(size = 0.02f)) shouldBe empty

  it should "not flag a legitimately large polytope as off-sphere or off-centre" in:
    PolytopeInvariants.check(Tesseract(size = 500f)) shouldBe empty

  it should "flag an Euler-Poincare violation when cell count is wrong" in:
    val brokenCells = Pentachoron().cells.drop(1) // 5 -> 4 cells, V-E+F-C != 0
    val findings = PolytopeInvariants.check(FakeMesh4D(cls = brokenCells))
    findings.map(_.invariant) should contain("euler-poincare")

  it should "not check Euler-Poincare when the mesh reports no cell structure" in:
    val findings = PolytopeInvariants.check(FakeMesh4D(cls = Seq.empty))
    findings.map(_.invariant) should not contain "euler-poincare"

  it should "report multiple independent findings when several invariants are violated" in:
    val verts = Pentachoron().vertices
    val withDuplicate = verts :+ verts(1) // duplicate an existing vertex
    val degenerateFace = Face4D[3](IndexedSeq(verts(0), verts(0), verts(0)))
    val findings = PolytopeInvariants.check(
      FakeMesh4D(verts = withDuplicate, fcs = Pentachoron().faces :+ degenerateFace)
    )
    findings.map(_.invariant) should contain allOf ("no-duplicate-vertices", "no-degenerate-faces")

  // A NaN/Inf coordinate makes every norm, centroid and area downstream of it meaningless, so
  // `check` reports it alone rather than emitting three more findings whose messages are all
  // NaN (review round 2).
  it should "report only the finite-coordinate finding when a coordinate is NaN" in:
    val verts = Pentachoron().vertices
    val corrupted = verts.updated(0, Vector[4](Float.NaN, 0f, 0f, 0f))
    val corruptedWithDuplicate = corrupted :+ corrupted(1)
    val findings = PolytopeInvariants.check(FakeMesh4D(verts = corruptedWithDuplicate))
    findings.map(_.invariant) shouldBe List("finite-vertex-coordinates")
