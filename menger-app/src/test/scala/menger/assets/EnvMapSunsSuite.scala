package menger.assets

import java.io.File
import java.nio.file.Files
import java.nio.file.Path

import menger.common.ConfigurationException
import menger.common.EnvironmentException
import org.scalatest.BeforeAndAfterAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EnvMapSunsSuite extends AnyFlatSpec with Matchers with BeforeAndAfterAll:

  private val resources = new File(sys.props("user.dir"), "src/test/resources").getPath
  private val sunset = "rogland_sunset_2k.hdr"
  private val validHalfAngles = Set(0.5f, 1f, 2f, 4f)
  private val tolerance = 1e-4f

  private val tempDir: Path = Files.createTempDirectory("env-map-suns-suite")
  private val notAMap = "not-a-map.hdr"
  Files.writeString(tempDir.resolve(notAMap), "this is not a Radiance file")

  override def afterAll(): Unit =
    Files.delete(tempDir.resolve(notAMap))
    Files.delete(tempDir)
    super.afterAll()

  "A lookup" should "measure the sun of a map under its texture directory" in:
    val sun = EnvMapSuns.Lookup(resources, strict = true).measure(sunset).get

    sun.directionToSun.magnitude shouldBe 1f +- tolerance
    validHalfAngles should contain(sun.halfAngleDegrees)
    sun.intensity should be > 0f
    List(sun.color.r, sun.color.g, sun.color.b).max shouldBe 1f +- tolerance

  it should "measure a file once and give the same measurement again" in:
    val lookup = EnvMapSuns.Lookup(resources, strict = true)

    lookup.measure(sunset).get shouldBe theSameInstanceAs(lookup.measure(sunset).get)

  it should "return None for a missing map when it is not strict" in:
    EnvMapSuns.Lookup(resources, strict = false).measure("no-such-map.hdr") shouldBe None

  it should "name where it looked for a missing map when it is strict" in:
    val problem = the[EnvironmentException] thrownBy
      EnvMapSuns.Lookup(resources, strict = true).measure("no-such-map.hdr")

    problem.getMessage should (include("no-such-map.hdr") and include("is not there"))

  it should "refuse a path that leaves the texture directory, in either mode" in:
    List(true, false).foreach: strict =>
      a[ConfigurationException] should be thrownBy
        EnvMapSuns.Lookup(resources, strict).measure("../escape.hdr")

  it should "refuse a file that is no Radiance map" in:
    val problem = the[ConfigurationException] thrownBy
      EnvMapSuns.Lookup(tempDir.toString, strict = false).measure(notAMap)

    problem.getMessage should include("cannot take the sun")
