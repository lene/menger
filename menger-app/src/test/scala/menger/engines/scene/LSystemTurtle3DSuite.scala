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
    // index 0 is the seed point at the turtle's starting position/width (Sprint 36 H3.6 --
    // runs are now anchored at their start position instead of only recording F-endpoints),
    // so widths(1)/widths(2) are the two F-drawn points, not widths(0)/widths(1).
    val widths = specs.head.curveData.get.widths
    widths.length should be >= 3
    if widths.length >= 3 then
      (widths(2) - widths(1) * 0.5f).abs should be < Tolerance

  "Segment accumulation" should "produce one spec for three consecutive Fs" in:
    val turtle = LSystemTurtle3D("FFF", 90f, 1.0f)
    val specs = turtle.generate()
    specs.length shouldBe 1
    // 4 points (seed + 3 F's), not 3 -- the seed anchors the run at the turtle's start position.
    specs.head.curveData.get.points.length shouldBe 12

  "Corner sharpening" should "triple the corner point at a 90-degree turn (Sprint 36 #17)" in:
    // Before the fix, every run rendered through a smooth cubic B-spline that rounds off sharp
    // turns -- fine for organic branches, wrong for a geometric curve like hilbert3d. A 90-degree
    // turn between two F's must now anchor the spline by tripling the corner control point.
    val turtle = LSystemTurtle3D("F+F", 90f, 1.0f)
    val specs = turtle.generate()
    specs.length shouldBe 1
    // 5 points (seed, corner x3, end), not 3 -- see CurveCornerSharpeningSuite for the
    // pure-function behavior this relies on.
    specs.head.curveData.get.points.length shouldBe 15

  "Gap handling" should "produce two specs for FFFFfFFFF" in:
    val turtle = LSystemTurtle3D("FFFFfFFFF", 90f, 1.0f)
    val specs = turtle.generate()
    specs.length shouldBe 2

  "Branch separation" should "produce 2 specs for F[+FFFF][-FFFF]" in:
    val turtle = LSystemTurtle3D("F[+FFFF][-FFFF]", 90f, 1.0f)
    val specs = turtle.generate()
    specs.length shouldBe 2

  "Single-F branches" should "not be silently dropped (Sprint 36 H3.6)" in:
    // Before the fix, a run's first point was only recorded by an F step, never seeded from
    // the turtle's position on entering `[` or after popping `]`. A branch containing exactly
    // one F (extremely common in real presets, e.g. fern3d's "F[&F]F[^F][&F]") then produced
    // a run of a single point, which emitRun silently drops (points.length < 2). Two adjacent
    // single-F branches with no F between them reproduces this exactly: the second branch's
    // run started from an empty (unseeded) point list.
    val turtle = LSystemTurtle3D("F[+F][-F]", 90f, 1.0f)
    val specs = turtle.generate()
    specs.length shouldBe 2

  "Roll (< and >)" should "rotate around the heading axis, not change heading direction" in:
    // A true roll rotates around the turtle's own heading, which is a fixed point of that
    // rotation -- two F's separated only by a roll must stay collinear. Before the fix, `<`/`>`
    // rotated around `state.up` (the same axis as `+`/`-`), so the roll silently turned the
    // turtle instead of rolling it, breaking collinearity.
    val turtle = LSystemTurtle3D("F<F", 45f, 1.0f, 1.0f, 1.0f, normalizeScale = false)
    val specs = turtle.generate()
    specs.length shouldBe 1
    val pts = specs.head.curveData.get.points.grouped(3).toVector
    pts.length shouldBe 3
    val dir1 = Vec3(pts(1)(0) - pts(0)(0), pts(1)(1) - pts(0)(1), pts(1)(2) - pts(0)(2))
    val dir2 = Vec3(pts(2)(0) - pts(1)(0), pts(2)(1) - pts(1)(1), pts(2)(2) - pts(1)(2))
    (dir1.x - dir2.x).abs should be < Tolerance
    (dir1.y - dir2.y).abs should be < Tolerance
    (dir1.z - dir2.z).abs should be < Tolerance

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
    // index 0 is the seed point at the turtle's starting position/width; see the analogous
    // comment on "Width scaling" above.
    val widths = specs.head.curveData.get.widths
    widths.length should be >= 3
    if widths.length >= 3 then
      (widths(2) - widths(1) * 0.5f).abs should be < Tolerance

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
