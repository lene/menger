package menger.cli

import menger.MengerCLIOptions

import org.rogach.scallop.exceptions.ScallopException
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CLIObjectOptionSuite extends AnyFlatSpec with Matchers:

  "--object" should "accept 'sphere' with --optix" in:
    val options = SafeMengerCLIOptions(Seq("--optix", "--object", "sphere"))
    options.objectType.toOption shouldBe Some("sphere")

  it should "accept 'cube' with --optix" in:
    val options = SafeMengerCLIOptions(Seq("--optix", "--object", "cube"))
    options.objectType.toOption shouldBe Some("cube")

  it should "reject without --optix" in:
    an[ScallopException] should be thrownBy:
      SafeMengerCLIOptions(Seq("--object", "sphere"))

  it should "reject unknown object type" in:
    an[ScallopException] should be thrownBy:
      SafeMengerCLIOptions(Seq("--optix", "--object", "pyramid"))

  "--optix" should "require --object option" in:
    an[ScallopException] should be thrownBy:
      SafeMengerCLIOptions(Seq("--optix"))

  it should "work with --object sphere" in:
    noException should be thrownBy:
      SafeMengerCLIOptions(Seq("--optix", "--object", "sphere"))

  it should "work with --object cube" in:
    noException should be thrownBy:
      SafeMengerCLIOptions(Seq("--optix", "--object", "cube"))
