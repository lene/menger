package menger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.common.Color

class ObjectSpecTest extends AnyFlatSpec with Matchers:

  "ObjectSpec.parse" should "parse sphere with minimal attributes" in:
    val result = ObjectSpec.parse("type=sphere")
    result shouldBe Right(ObjectSpec("sphere", 0.0f, 0.0f, 0.0f, 1.0f, None, None, 1.0f))

  it should "parse sphere with all attributes" in:
    val result = ObjectSpec.parse("type=sphere:pos=1,2,3:size=2.5:color=#FF0000:ior=1.5")
    result match
      case Right(spec) =>
        spec.objectType shouldBe "sphere"
        spec.x shouldBe 1.0f
        spec.y shouldBe 2.0f
        spec.z shouldBe 3.0f
        spec.size shouldBe 2.5f
        spec.color.get.r shouldBe 1.0f
        spec.color.get.g shouldBe 0.0f
        spec.color.get.b shouldBe 0.0f
        spec.ior shouldBe 1.5f
      case Left(error) => fail(s"Expected Right but got Left: $error")

  it should "parse cube with position and size" in:
    val result = ObjectSpec.parse("type=cube:pos=-1,0,1:size=3.0")
    result match
      case Right(spec) =>
        spec.objectType shouldBe "cube"
        spec.x shouldBe -1.0f
        spec.y shouldBe 0.0f
        spec.z shouldBe 1.0f
        spec.size shouldBe 3.0f
      case Left(error) => fail(s"Expected Right but got Left: $error")

  it should "parse sponge-volume with level" in:
    val result = ObjectSpec.parse("type=sponge-volume:level=3:size=2.0")
    result match
      case Right(spec) =>
        spec.objectType shouldBe "sponge-volume"
        spec.level shouldBe Some(3.0f)
        spec.size shouldBe 2.0f
      case Left(error) => fail(s"Expected Right but got Left: $error")

  it should "parse sponge-surface with fractional level" in:
    val result = ObjectSpec.parse("type=sponge-surface:level=2.5:pos=0,1,0:color=#00FF00")
    result match
      case Right(spec) =>
        spec.objectType shouldBe "sponge-surface"
        spec.level shouldBe Some(2.5f)
        spec.x shouldBe 0.0f
        spec.y shouldBe 1.0f
        spec.z shouldBe 0.0f
        spec.color.get.g shouldBe 1.0f
      case Left(error) => fail(s"Expected Right but got Left: $error")

  it should "handle color without # prefix" in:
    val result = ObjectSpec.parse("type=sphere:color=FF0000")
    result match
      case Right(spec) =>
        spec.color.get.r shouldBe 1.0f
        spec.color.get.g shouldBe 0.0f
        spec.color.get.b shouldBe 0.0f
      case Left(error) => fail(s"Expected Right but got Left: $error")

  it should "handle whitespace in spec" in:
    val result = ObjectSpec.parse("type = sphere : pos = 1, 2, 3 : size = 1.5")
    result match
      case Right(spec) =>
        spec.objectType shouldBe "sphere"
        spec.x shouldBe 1.0f
        spec.y shouldBe 2.0f
        spec.z shouldBe 3.0f
        spec.size shouldBe 1.5f
      case Left(error) => fail(s"Expected Right but got Left: $error")

  it should "fail when type is missing" in:
    val result = ObjectSpec.parse("pos=1,2,3:size=2.0")
    result shouldBe a[Left[_, _]]
    result.left.map(_ should include("Missing required 'type' field"))

  it should "fail when type is invalid" in:
    val result = ObjectSpec.parse("type=pyramid:pos=0,0,0")
    result shouldBe a[Left[_, _]]
    result.left.map(_ should include("Invalid object type"))

  it should "fail when position format is invalid" in:
    val result = ObjectSpec.parse("type=sphere:pos=1,2")
    result shouldBe a[Left[_, _]]
    result.left.map(_ should include("Invalid position format"))

  it should "fail when sponge is missing level" in:
    val result = ObjectSpec.parse("type=sponge-volume:size=2.0")
    result shouldBe a[Left[_, _]]
    result.left.map(_ should include("requires 'level' field"))

  it should "accept sponge-volume type (case insensitive)" in:
    val result = ObjectSpec.parse("type=SPONGE-VOLUME:level=2")
    result match
      case Right(spec) => spec.objectType shouldBe "sponge-volume"
      case Left(error) => fail(s"Expected Right but got Left: $error")

  "ObjectSpec.parseAll" should "parse multiple valid specs" in:
    val specs = List(
      "type=sphere:pos=0,0,0:size=1.0",
      "type=cube:pos=2,0,0:size=1.5",
      "type=sponge-volume:pos=-2,0,0:level=2:size=2.0"
    )
    val result = ObjectSpec.parseAll(specs)
    result match
      case Right(objects) =>
        objects.length shouldBe 3
        objects(0).objectType shouldBe "sphere"
        objects(1).objectType shouldBe "cube"
        objects(2).objectType shouldBe "sponge-volume"
      case Left(error) => fail(s"Expected Right but got Left: $error")

  it should "fail if any spec is invalid" in:
    val specs = List(
      "type=sphere:pos=0,0,0",
      "type=invalid:pos=1,1,1",  // Invalid type
      "type=cube:pos=2,0,0"
    )
    val result = ObjectSpec.parseAll(specs)
    result shouldBe a[Left[_, _]]

  it should "return empty list for empty input" in:
    val result = ObjectSpec.parseAll(List.empty)
    result shouldBe Right(List.empty)

  "Complex object specs" should "support all sphere attributes" in:
    val spec = "type=sphere:pos=1.5,-2.3,4.7:size=3.2:color=#8080FF:ior=2.42"
    val result = ObjectSpec.parse(spec)
    result match
      case Right(obj) =>
        obj.objectType shouldBe "sphere"
        obj.x shouldBe 1.5f
        obj.y shouldBe -2.3f
        obj.z shouldBe 4.7f
        obj.size shouldBe 3.2f
        obj.ior shouldBe 2.42f
        obj.color.isDefined shouldBe true
      case Left(error) => fail(s"Expected Right but got Left: $error")

  it should "support sponge with color and ior" in:
    val spec = "type=sponge-surface:level=1.5:pos=0,0,0:size=2.0:color=#FF00FF:ior=1.5"
    val result = ObjectSpec.parse(spec)
    result match
      case Right(obj) =>
        obj.objectType shouldBe "sponge-surface"
        obj.level shouldBe Some(1.5f)
        obj.color.isDefined shouldBe true
        obj.ior shouldBe 1.5f
      case Left(error) => fail(s"Expected Right but got Left: $error")
