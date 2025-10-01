package menger.objects

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CompositeSuite extends AnyFlatSpec with Matchers:

  "Composite toString" should "show component geometries" in:
    val sphere = Sphere()
    val cube = Cube()
    val composite = Composite(geometries = List(sphere, cube))
    composite.toString should be("Composite(Sphere, Cube)")

  "Composite.isValidSpongeType" should "accept basic 3D shapes" in:
    Composite.isValidSpongeType("cube") should be(true)
    Composite.isValidSpongeType("square") should be(true)

  it should "accept 4D shapes when standalone" in:
    Composite.isValidSpongeType("tesseract") should be(true)
    Composite.isValidSpongeType("tesseract-sponge") should be(true)
    Composite.isValidSpongeType("tesseract-sponge-2") should be(true)

  it should "accept new sponge types" in:
    Composite.isValidSpongeType("square-sponge") should be(true)
    Composite.isValidSpongeType("cube-sponge") should be(true)

  it should "accept simple composites with 3D shapes" in:
    Composite.isValidSpongeType("composite[cube,square]") should be(true)
    Composite.isValidSpongeType("composite[cube]") should be(true)

  it should "reject composites with 4D shapes" in:
    Composite.isValidSpongeType("composite[cube,tesseract]") should be(false)
    Composite.isValidSpongeType("composite[tesseract]") should be(false)
    Composite.isValidSpongeType("composite[tesseract-sponge]") should be(false)

  it should "accept nested composites with only 3D shapes" in:
    Composite.isValidSpongeType("composite[composite[cube,square],cube]") should be(true)
    Composite.isValidSpongeType("composite[cube,composite[square]]") should be(true)

  it should "reject nested composites containing 4D shapes" in:
    Composite.isValidSpongeType("composite[composite[cube,tesseract],cube]") should be(false)
    Composite.isValidSpongeType("composite[cube,composite[tesseract]]") should be(false)

  it should "reject invalid sponge types" in:
    Composite.isValidSpongeType("invalid") should be(false)
    Composite.isValidSpongeType("composite[invalid]") should be(false)
    Composite.isValidSpongeType("composite[]") should be(false)

  it should "reject malformed composite syntax" in:
    Composite.isValidSpongeType("composite[cube") should be(false)
    Composite.isValidSpongeType("compositecube,square]") should be(false)
    Composite.isValidSpongeType("composite") should be(false)
