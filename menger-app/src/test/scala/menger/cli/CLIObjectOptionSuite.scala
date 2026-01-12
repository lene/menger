package menger.cli

import org.rogach.scallop.exceptions.ScallopException
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CLIObjectOptionSuite extends AnyFlatSpec with Matchers:

  "--objects" should "accept 'type=sphere' with --optix" in:
    val options = SafeMengerCLIOptions(Seq("--optix", "--objects", "type=sphere"))
    options.objects.toOption.flatMap(_.headOption.map(_.objectType)) shouldBe Some("sphere")

  it should "accept 'type=cube' with --optix" in:
    val options = SafeMengerCLIOptions(Seq("--optix", "--objects", "type=cube"))
    options.objects.toOption.flatMap(_.headOption.map(_.objectType)) shouldBe Some("cube")

  it should "reject without --optix" in:
    an[ScallopException] should be thrownBy:
      SafeMengerCLIOptions(Seq("--objects", "type=sphere"))

  it should "reject unknown object type" in:
    an[ScallopException] should be thrownBy:
      SafeMengerCLIOptions(Seq("--optix", "--objects", "type=pyramid"))

  "--optix" should "require --objects option" in:
    an[ScallopException] should be thrownBy:
      SafeMengerCLIOptions(Seq("--optix"))

  it should "work with --objects type=sphere" in:
    noException should be thrownBy:
      SafeMengerCLIOptions(Seq("--optix", "--objects", "type=sphere"))

  it should "work with --objects type=cube" in:
    noException should be thrownBy:
      SafeMengerCLIOptions(Seq("--optix", "--objects", "type=cube"))
