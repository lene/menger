package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.language.implicitConversions

class SceneObjectSuite extends AnyFlatSpec with Matchers:

  "Sphere" should "be constructible with defaults" in:
    val sphere = Sphere()
    sphere.pos shouldBe Vec3.Zero
    sphere.size shouldBe 1.0f
    sphere.material shouldBe None
    sphere.color shouldBe None

  it should "be constructible with material only" in:
    val sphere = Sphere(Material.Glass)
    sphere.pos shouldBe Vec3.Zero
    sphere.material shouldBe Some(Material.Glass)

  it should "be constructible with position and material" in:
    val sphere = Sphere(Vec3(1f, 2f, 3f), Material.Gold)
    sphere.pos shouldBe Vec3(1f, 2f, 3f)
    sphere.material shouldBe Some(Material.Gold)

  it should "be constructible with position, material, and size" in:
    val sphere = Sphere(Vec3(1f, 2f, 3f), Material.Chrome, 0.5f)
    sphere.size shouldBe 0.5f

  it should "accept Float tuple positions" in:
    val sphere = Sphere((1f, 2f, 3f), Material.Glass)
    sphere.pos shouldBe Vec3(1f, 2f, 3f)

  it should "accept Int tuple positions" in:
    val sphere = Sphere((1, 2, 3), Material.Glass)
    sphere.pos shouldBe Vec3(1f, 2f, 3f)

  it should "accept Double tuple positions" in:
    val sphere = Sphere((1.0, 2.0, 3.0), Material.Glass)
    sphere.pos shouldBe Vec3(1f, 2f, 3f)

  it should "validate positive size" in:
    an[IllegalArgumentException] should be thrownBy Sphere(size = 0f)
    an[IllegalArgumentException] should be thrownBy Sphere(size = -1f)

  it should "validate non-negative IOR" in:
    an[IllegalArgumentException] should be thrownBy Sphere(ior = -0.1f)

  "Sphere.toObjectSpec" should "create correct ObjectSpec" in:
    val sphere = Sphere(Vec3(1f, 2f, 3f), Material.Glass, 0.5f)
    val spec = sphere.toObjectSpec

    spec.objectType shouldBe "sphere"
    spec.x shouldBe 1f
    spec.y shouldBe 2f
    spec.z shouldBe 3f
    spec.size shouldBe 0.5f
    spec.level shouldBe None
    spec.material shouldBe Some(Material.Glass.toOptixMaterial)

  it should "include color when specified" in:
    val sphere = Sphere(color = Some(Color.Red))
    val spec = sphere.toObjectSpec
    spec.color shouldBe Some(Color.Red.toCommonColor)

  it should "include texture when specified" in:
    val sphere = Sphere(texture = Some("brick.png"))
    val spec = sphere.toObjectSpec
    spec.texture shouldBe Some("brick.png")

  "Cube" should "be constructible with defaults" in:
    val cube = Cube()
    cube.pos shouldBe Vec3.Zero
    cube.size shouldBe 1.0f

  it should "be constructible with material only" in:
    val cube = Cube(Material.Gold)
    cube.material shouldBe Some(Material.Gold)

  it should "be constructible with position and material" in:
    val cube = Cube(Vec3(-2f, 0f, 0f), Material.Copper)
    cube.pos shouldBe Vec3(-2f, 0f, 0f)
    cube.material shouldBe Some(Material.Copper)

  it should "accept tuple positions" in:
    val cube = Cube((1, 2, 3), Material.Chrome)
    cube.pos shouldBe Vec3(1f, 2f, 3f)

  it should "validate positive size" in:
    an[IllegalArgumentException] should be thrownBy Cube(size = 0f)

  "Cube.toObjectSpec" should "create correct ObjectSpec" in:
    val cube = Cube(Vec3(-2f, 0f, 0f), Material.Copper, 1.5f)
    val spec = cube.toObjectSpec

    spec.objectType shouldBe "cube"
    spec.x shouldBe -2f
    spec.y shouldBe 0f
    spec.z shouldBe 0f
    spec.size shouldBe 1.5f

  "SpongeType" should "have correct type names" in:
    SpongeType.VolumeFilling.objectTypeName shouldBe "sponge-volume"
    SpongeType.SurfaceUnfolding.objectTypeName shouldBe "sponge-surface"
    SpongeType.CubeSponge.objectTypeName shouldBe "cube-sponge"

  "Sponge" should "be constructible with type and level" in:
    val sponge = Sponge(VolumeFilling, level = 2f)
    sponge.spongeType shouldBe VolumeFilling
    sponge.level shouldBe 2f
    sponge.pos shouldBe Vec3.Zero

  it should "be constructible with type, level, and material" in:
    val sponge = Sponge(SurfaceUnfolding, level = 3f, Material.Glass)
    sponge.material shouldBe Some(Material.Glass)

  it should "be constructible with type, level, material, and size" in:
    val sponge = Sponge(VolumeFilling, level = 2f, Material.Glass, size = 2.0f)
    sponge.size shouldBe 2.0f

  it should "be constructible with position" in:
    val sponge = Sponge(Vec3(4f, 0f, 0f), SurfaceUnfolding, level = 2f)
    sponge.pos shouldBe Vec3(4f, 0f, 0f)

  it should "be constructible with position and material" in:
    val sponge = Sponge(Vec3(4f, 0f, 0f), VolumeFilling, level = 2f, Material.Glass)
    sponge.pos shouldBe Vec3(4f, 0f, 0f)
    sponge.material shouldBe Some(Material.Glass)

  it should "accept Float tuple positions" in:
    val sponge = Sponge((4f, 0f, 0f), CubeSponge, level = 1f)
    sponge.pos shouldBe Vec3(4f, 0f, 0f)

  it should "accept Int tuple positions" in:
    val sponge = Sponge((4, 0, 0), CubeSponge, level = 1f)
    sponge.pos shouldBe Vec3(4f, 0f, 0f)

  it should "validate non-negative level" in:
    an[IllegalArgumentException] should be thrownBy Sponge(VolumeFilling, level = -1f)

  it should "validate positive size" in:
    an[IllegalArgumentException] should be thrownBy Sponge(VolumeFilling, level = 2f, size = 0f)

  "Sponge.toObjectSpec" should "create correct ObjectSpec" in:
    val sponge = Sponge(Vec3(4f, 0f, 0f), SurfaceUnfolding, level = 2f, Material.Glass, size = 2.0f)
    val spec = sponge.toObjectSpec

    spec.objectType shouldBe "sponge-surface"
    spec.x shouldBe 4f
    spec.y shouldBe 0f
    spec.z shouldBe 0f
    spec.size shouldBe 2.0f
    spec.level shouldBe Some(2f)
    spec.material shouldBe Some(Material.Glass.toOptixMaterial)

  it should "normalize deprecated sponge type names" in:
    val sponge = Sponge(VolumeFilling, level = 2f)
    val spec = sponge.toObjectSpec
    // VolumeFilling maps to "sponge-volume", which is already canonical
    spec.objectType shouldBe "sponge-volume"

  it should "include color when specified" in:
    val sponge = Sponge(VolumeFilling, level = 2f, color = Some(Color("#00FF00")))
    val spec = sponge.toObjectSpec
    spec.color shouldBe Some(Color("#00FF00").toCommonColor)
