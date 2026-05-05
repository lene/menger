package menger.objects.higher_d

import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Validates CellShapeContract.assertDodecahedron against a known-correct fixture.
  * "Validate the detector before trusting it" — per debugging-rendering-bugs skill. */
class DodecahedronContractSpec extends AnyFlatSpec with Matchers:

  private val phi = ((1 + math.sqrt(5)) / 2).toFloat
  private val inv = (1.0 / phi).toFloat

  // Standard golden-ratio dodecahedron, embedded in 4D with w=0.
  // 8 cubic + 4 + 4 + 4 rectangular-face vertices = 20 total.
  private val fixture: CellShapeContract.Cell4D =
    (for sa <- Seq(-1f, 1f); sb <- Seq(-1f, 1f); sc <- Seq(-1f, 1f) yield
      Vector[4](sa, sb, sc, 0f)) ++
    (for sb <- Seq(-1f, 1f); sc <- Seq(-1f, 1f) yield
      Vector[4](0f, sb * inv, sc * phi, 0f)) ++
    (for sa <- Seq(-1f, 1f); sc <- Seq(-1f, 1f) yield
      Vector[4](sa * inv, sc * phi, 0f, 0f)) ++
    (for sa <- Seq(-1f, 1f); sb <- Seq(-1f, 1f) yield
      Vector[4](sa * phi, 0f, sb * inv, 0f))

  "DodecahedronContractSpec fixture" should "have 20 vertices" in:
    fixture should have size 20

  it should "pass assertDodecahedron" in:
    noException should be thrownBy CellShapeContract.assertDodecahedron(fixture)

  it should "fail assertTetrahedron (wrong vertex count)" in:
    an[AssertionError] should be thrownBy CellShapeContract.assertTetrahedron(fixture)

  it should "fail assertOctahedron (wrong vertex count)" in:
    an[AssertionError] should be thrownBy CellShapeContract.assertOctahedron(fixture)
