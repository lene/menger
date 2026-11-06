package menger.objects.higher_d

import menger.PropertyTestGenerators.dimensions
import menger.PropertyTestGenerators.vectorGen
import menger.common.Vector
import menger.objects.higher_d.CustomMatchers.closeTo
import org.scalacheck.Gen
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class PerspectiveChainSuite extends AnyFlatSpec with Matchers with ScalaCheckPropertyChecks:

  private val clipEpsilon = 1e-6f
  private val absoluteTolerance = 5e-4f

  private val stageGen: Gen[Projection] = for
    screen <- Gen.choose(0.5f, 2f)
    gap <- Gen.choose(0.5f, 5f)
  yield Projection(screen + gap, screen)

  private def stagesGen(d: Dimension): Gen[Seq[Projection]] =
    Gen.listOfN(d.value - 3, stageGen)

  // coordinates in (-1, 1) keep every stage's dropped coordinate below its eye (eye >= 1)
  private def nearVectorGen(d: Dimension): Gen[Vector[d.N]] =
    vectorGen(d, Gen.choose(-0.9f, 0.9f))

  private def bits(values: Iterable[Float]): Seq[Int] =
    values.map(java.lang.Float.floatToRawIntBits).toSeq

  // Independent formulation: recursion from the highest dropped axis down.
  private def expected(stages: Seq[Projection], coordinates: Seq[Float]): Seq[Float] =
    stages match
      case Seq() => coordinates
      case stage +: rest =>
        val factor = (stage.eyeW - stage.screenW) / (stage.eyeW - coordinates.last)
        expected(rest, coordinates.init.map(_ * factor))

  "A chain without stages (N = 3)" should "leave a point bit for bit as it is" in:
    forAll(Gen.listOfN(3, Gen.choose(-100f, 100f))) { xs =>
      val p = Vector[3](xs*)
      bits(PerspectiveChain[3](Seq.empty)(p).toIndexedSeq) shouldBe bits(xs)
      PerspectiveChain[3](Seq.empty).project(p) shouldBe Some(p)
    }

  "A chain with one stage (N = 4)" should "equal Projection.apply bit for bit" in:
    forAll(stageGen, Gen.listOfN(4, Gen.choose(-0.9f, 0.9f))) { (stage, xs) =>
      val p = Vector[4](xs*)
      bits(PerspectiveChain[4](Seq(stage))(p).toIndexedSeq) shouldBe
        bits(stage(p).toIndexedSeq)
    }

  "A chain" should "follow each stage's own (eye, screen) pair, first stage first" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      forAll(stagesGen(d), nearVectorGen(d)) { (stages, p) =>
        val chain = PerspectiveChain[d.N](stages)
        bits(chain(p).toIndexedSeq) shouldBe bits(expected(stages, p.toIndexedSeq))
      }
    }

  it should "scale a flat point by the product of the stage factors" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      forAll(stagesGen(d), Gen.listOfN(3, Gen.choose(-5f, 5f))) { (stages, xs) =>
        val flat = Vector.fromSeq[d.N](xs ++ Seq.fill(d.value - 3)(0f))
        val scale = stages.map(s => (s.eyeW - s.screenW) / s.eyeW).product
        val projected = PerspectiveChain[d.N](stages)(flat)
        projected should closeTo(Vector[3](xs.map(_ * scale)*), absoluteTolerance)
      }
    }

  it should "give a different projection when the stage pairs are swapped" in:
    val (a, b) = (Projection(3f, 1f), Projection(6f, 2f))
    val p = Vector[5](0.5f, 1f, 1.5f, 0.8f, 0.9f)
    PerspectiveChain[5](Seq(a, b))(p) should not be closeTo(PerspectiveChain[5](Seq(b, a))(p), 1e-3f)

  it should "project to None exactly when a stage's dropped coordinate reaches its eye" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      val farGen = vectorGen(d, Gen.choose(-4f, 8f))
      forAll(stagesGen(d), farGen) { (stages, p) =>
        val clipped = stages.foldLeft((p.toIndexedSeq, false)) {
          case ((coordinates, alreadyClipped), stage) =>
            val factor = (stage.eyeW - stage.screenW) / (stage.eyeW - coordinates.last)
            (
              coordinates.init.map(_ * factor),
              alreadyClipped || coordinates.last >= stage.eyeW - clipEpsilon
            )
        }._2
        PerspectiveChain[d.N](stages).project(p).isEmpty shouldBe clipped
      }
    }

  it should "project a point at, just behind and just before the eye as the clip rule says" in:
    val eye = 3f
    val chain = PerspectiveChain[4](Seq(Projection(eye, 1f)))
    def at(w: Float) = Vector[4](1f, 1f, 1f, w)
    chain.project(at(eye)) shouldBe None
    chain.project(at(eye + 1f)) shouldBe None
    chain.project(at(eye - clipEpsilon)) shouldBe None
    chain.project(at(eye - 1e-3f)).isDefined shouldBe true
    chain.project(at(0f)) shouldBe Some(chain(at(0f)))

  it should "clip a later stage on the already scaled coordinate" in:
    val chain = PerspectiveChain[5](Seq(Projection(2f, 1f), Projection(2f, 1f)))
    // stage 1 factor is 1 / (2 - 0) = 0.5, so w becomes 1.5 (< 2): not clipped
    chain.project(Vector[5](1f, 1f, 1f, 3f, 0f)).isDefined shouldBe true
    // stage 1 factor is 1 / (2 - 1.5) = 2, so w becomes 8 (>= 2): clipped
    chain.project(Vector[5](1f, 1f, 1f, 4f, 1.5f)) shouldBe None

  "Constructing a chain" should "reject a wrong stage count, naming both counts" in:
    val tooFew = the[IllegalArgumentException] thrownBy
      PerspectiveChain[6](Seq(Projection(3f, 1f)))
    tooFew.getMessage should (include("needs 3 stages") and include("got 1") and include("6D"))

  it should "reject any stage for N = 3, naming both counts" in:
    val tooMany = the[IllegalArgumentException] thrownBy
      PerspectiveChain[3](Seq(Projection(3f, 1f)))
    tooMany.getMessage should (include("needs 0 stages") and include("got 1"))

  it should "reject a dimension outside 3..8" in:
    val unsupported = the[IllegalArgumentException] thrownBy
      PerspectiveChain[9](Seq.fill(6)(Projection(3f, 1f)))
    unsupported.getMessage should include("9")

  "PerspectiveChain.uniform" should "use one (eye, screen) pair for all N - 3 stages" in:
    dimensions.foreach { d =>
      given ValueOf[d.N] = d.valueOf
      PerspectiveChain.uniform[d.N](3f, 1f).stages shouldBe Seq.fill(d.value - 3)(Projection(3f, 1f))
    }

  "Projection.factor" should "be the stage factor Projection.apply uses" in:
    val projection = Projection(3f, 1f)
    projection.factor(0f) shouldBe (2f / 3f)
    projection.factor(2f) shouldBe 2f
