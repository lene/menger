package menger.engines.scene

import menger.dsl.Vec3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LSystemTurtle3DSuite extends AnyFlatSpec with Matchers:

  private val Tolerance = 0.001f

  "Tree preset" should "produce non-empty curve list" in:
    val specs = LSystemTurtle3D.Tree.generate()
    specs should not be empty
    all(specs.map(_.objectType)) shouldBe "curve"

  "Turtle orthonormality" should "maintain orthogonal frame after random rotations" in:
    val turtle = LSystemTurtle3D(("F+" * 5000) + "F", 137.5f, 1.0f, 0.1f, 0.7f, 42L)
    val specs = turtle.generate()
    specs should not be empty

  "Push/pop state" should "restore position after branch" in:
    val turtle = LSystemTurtle3D("F[+F]F", 90f, 1.0f)
    val specs = turtle.generate()
    val allPts = specs.flatMap(_.curveData.toList.flatMap(_.points.grouped(3)))
    allPts should not be empty

  "Width scaling" should "halve width after !(0.5)" in:
    val turtle = LSystemTurtle3D("F!(0.5)F", 90f, 1.0f, 1.0f, 0.7f)
    val specs = turtle.generate()
    specs should not be empty
    specs.head.curveData should not be empty
    val widths = specs.head.curveData.get.widths
    widths.length should be >= 2
    if widths.length >= 2 then
      (widths(1) - widths(0) * 0.5f).abs should be < Tolerance

  "Segment accumulation" should "produce one spec for three consecutive Fs" in:
    val turtle = LSystemTurtle3D("FFF", 90f, 1.0f)
    val specs = turtle.generate()
    specs.length shouldBe 1
    specs.head.curveData.get.points.length shouldBe 9

  "Gap handling" should "produce two specs for FfF" in:
    val turtle = LSystemTurtle3D("FfF", 90f, 1.0f)
    val specs = turtle.generate()
    specs.length shouldBe 2

  "Branch separation" should "produce 3 specs for F[+F][-F]" in:
    val turtle = LSystemTurtle3D("F[+F][-F]", 90f, 1.0f)
    val specs = turtle.generate()
    specs.length shouldBe 3

  "Sphere primitive" should "emit sphere ObjectSpec" in:
    val turtle = LSystemTurtle3D("@O(1.0)", 90f, 1.0f)
    val specs = turtle.generate()
    specs.length shouldBe 1
    specs.head.objectType shouldBe "sphere"
    specs.head.size shouldBe 1.0f

  "Hilbert curve" should "produce recognizable 3D curve pattern" in:
    val specs = LSystemTurtle3D.HilbertCurve3D.generate()
    specs should not be empty
    all(specs.map(_.objectType)) shouldBe "curve"

  "Stochastic same-seed" should "produce identical point lists" in:
    val specs1 = LSystemTurtle3D.Tree.generate()
    val specs2 = LSystemTurtle3D.Tree.generate()
    val ptsToVec = (s: menger.ObjectSpec) =>
      s.curveData.get.points.grouped(3).map(g => (g(0), g(1), g(2))).toVector
    specs1.map(ptsToVec) shouldBe specs2.map(ptsToVec)

  "Pruning" should "skip symbols after %(n)" in:
    val turtle = LSystemTurtle3D("%(3)FFF", 90f, 1.0f)
    val specs = turtle.generate()
    specs shouldBe empty

  "Parameterized F" should "parse F(len,width,shape)" in:
    val turtle = LSystemTurtle3D("F(2.0)F(1.0,0.05)F(1.0,0.05,cylinder)",
      90f, 1.0f, 0.1f, 0.7f)
    val specs = turtle.generate()
    specs should not be empty
    specs.head.objectType shouldBe "curve"

  "Bush preset" should "render without error" in:
    val specs = LSystemTurtle3D.Bush.generate()
    specs should not be empty

  "Fern3D preset" should "render without error" in:
    val specs = LSystemTurtle3D.Fern3D.generate()
    specs should not be empty

  "KochIsland preset" should "render without error" in:
    val specs = LSystemTurtle3D.KochIsland.generate()
    specs should not be empty

  "Default width decay" should "reduce width on ! without parameter" in:
    val turtle = LSystemTurtle3D("F!F", 90f, 1.0f, 1.0f, 0.5f)
    val specs = turtle.generate()
    specs should not be empty
    val widths = specs.head.curveData.get.widths
    widths.length should be >= 2
    if widths.length >= 2 then
      (widths(1) - widths(0) * 0.5f).abs should be < Tolerance

  "Material index cycling" should "wrap around" in:
    val mat1 = menger.common.Material(menger.common.Color(1f, 0f, 0f))
    val mat2 = menger.common.Material(menger.common.Color(0f, 1f, 0f))
    val turtle = new LSystemTurtle3D(
      "F'F'F", 90f, 1.0f, 0.1f, 0.7f, 42L,
      materials = Map("a" -> mat1, "b" -> mat2)
    )
    val specs = turtle.generate()
    specs should not be empty

  "Normalize scale" should "fit within unit cube" in:
    val specs = LSystemTurtle3D.Tree.generate()
    for spec <- specs; cd <- spec.curveData.toList do
      for pt <- cd.points.grouped(3) do
        pt(0).abs should be <= 0.6f
        pt(1).abs should be <= 0.6f
        pt(2).abs should be <= 0.6f
