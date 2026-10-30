package menger

import com.tngtech.archunit.core.importer.ClassFileImporter
import menger.ArchitecturePhase2Spec.AllowedLayerDependencies
import menger.ArchitecturePhase2Spec.layerRule
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** menger#90: eight architecture checks "cannot fail as written", and a passing vacuous rule
  * reads as coverage. This shows each layer's allow-list rule going red on a class that depends on
  * a layer it may not (PlantedViolations), and a rule for a package that does not exist going red
  * instead of passing on nothing. That the same rules pass on the real classes is
  * ArchitecturePhase2Spec. */
class ArchitectureRulesCanFailSpec extends AnyFlatSpec with Matchers:

  for (layer, allowed) <- AllowedLayerDependencies do
    s"The $layer rule" should "fail on a class that depends on a layer it may not" in:
      val planted = ClassFileImporter().importPackages(s"$layer.planted")

      planted.size should be > 0
      layerRule(layer, allowed).evaluate(planted).hasViolation shouldBe true

  "A layer rule for a package with no classes" should "fail instead of passing on nothing" in:
    val someClasses = ClassFileImporter().importPackages("menger.common")

    an[AssertionError] should be thrownBy layerRule("menger.nonexistent", Nil).check(someClasses)
