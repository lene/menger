package menger.cli

import org.rogach.scallop.exceptions.ScallopException
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CLIObjectOptionSuite extends AnyFlatSpec with Matchers:

  "--objects" should "accept 'type=sphere'" in:
    val options = SafeMengerCLIOptions(Seq("--objects", "type=sphere"))
    options.objects.toOption.flatMap(_.headOption.map(_.objectType)) shouldBe Some("sphere")

  it should "accept 'type=cube'" in:
    val options = SafeMengerCLIOptions(Seq("--objects", "type=cube"))
    options.objects.toOption.flatMap(_.headOption.map(_.objectType)) shouldBe Some("cube")

  it should "reject unknown object type" in:
    an[ScallopException] should be thrownBy:
      SafeMengerCLIOptions(Seq("--objects", "type=pyramid"))

  "--objects type=sphere" should "work standalone" in:
    noException should be thrownBy:
      SafeMengerCLIOptions(Seq("--objects", "type=sphere"))

  it should "work with type=cube" in:
    noException should be thrownBy:
      SafeMengerCLIOptions(Seq("--objects", "type=cube"))
