package menger.dsl

import scala.language.implicitConversions

import menger.spec.ObjectType
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

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
    spec.material shouldBe Some(Material.Glass.toCoreMaterial)

  it should "include color when specified" in:
    val sphere = Sphere(color = Some(Color.Red))
    val spec = sphere.toObjectSpec
    spec.color shouldBe Some(Color.Red.toCommonColor)

  it should "include texture when specified" in:
    val sphere = Sphere(texture = Some("brick.png"))
    val spec = sphere.toObjectSpec
    spec.texture shouldBe Some("brick.png")

  it should "include video texture when specified" in:
    val videoTexture = VideoTexture("clips/checker.mov")
    val sphere = Sphere(videoTexture = Some(videoTexture))
    val spec = sphere.toObjectSpec

    spec.videoTexture shouldBe Some(videoTexture)
    spec.texture shouldBe None

  it should "reject objects with both image and video textures" in:
    val sphere = Sphere(
      texture = Some("brick.png"),
      videoTexture = Some(VideoTexture("clips/checker.mov"))
    )

    an[IllegalArgumentException] should be thrownBy sphere.toObjectSpec

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

  // Usability review 2026-09 (T1#3): the DSL had no upper bound at all -- only the CLI path did.
  it should "reject a level above the hard maximum" in:
    an[IllegalArgumentException] should be thrownBy
      Sponge(VolumeFilling, level = (ResourceLimits.volumeSpongeLevel.max + 1).toFloat)

  it should "accept a level at the hard maximum" in:
    noException should be thrownBy Sponge(VolumeFilling, level = ResourceLimits.volumeSpongeLevel.max.toFloat)

  // Usability review 2026-10, session 4 (F86): one limit per sponge type; CubeSponge had no DSL
  // limit at all although its instances run out above level 3.
  it should "apply each sponge type's own level maximum" in:
    noException should be thrownBy Sponge(VolumeFilling, level = 5f)
    an[IllegalArgumentException] should be thrownBy Sponge(VolumeFilling, level = 5.5f)
    noException should be thrownBy Sponge(SurfaceUnfolding, level = 5f)
    noException should be thrownBy Sponge(CubeSponge, level = 3f)
    an[IllegalArgumentException] should be thrownBy Sponge(CubeSponge, level = 3.5f)

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
    spec.material shouldBe Some(Material.Glass.toCoreMaterial)

  it should "normalize deprecated sponge type names" in:
    val sponge = Sponge(VolumeFilling, level = 2f)
    val spec = sponge.toObjectSpec
    // VolumeFilling maps to "sponge-volume", which is already canonical
    spec.objectType shouldBe "sponge-volume"

  it should "include color when specified" in:
    val sponge = Sponge(VolumeFilling, level = 2f, color = Some(Color("#00FF00")))
    val spec = sponge.toObjectSpec
    spec.color shouldBe Some(Color("#00FF00").toCommonColor)

  "VideoTexture" should "be supported by every textured DSL object" in:
    val videoTexture = VideoTexture("clips/checker.mov")
    val objects = List(
      Sphere(videoTexture = Some(videoTexture)),
      Cube(videoTexture = Some(videoTexture)),
      Sponge(VolumeFilling, level = 1f, videoTexture = Some(videoTexture)),
      Tesseract(videoTexture = Some(videoTexture)),
      TesseractSponge(VolumeRemoving, level = 1f, videoTexture = Some(videoTexture)),
      PentachoronGasket(level = 1f, videoTexture = Some(videoTexture)),
      Menger4D(level = 1f, videoTexture = Some(videoTexture)),
      HexadecachoronGasket(level = 1f, videoTexture = Some(videoTexture)),
      Tetrahedron(videoTexture = Some(videoTexture)),
      Octahedron(videoTexture = Some(videoTexture)),
      Dodecahedron(videoTexture = Some(videoTexture)),
      Icosahedron(videoTexture = Some(videoTexture)),
      Cone(videoTexture = Some(videoTexture)),
      ParametricSurface(
        f = (_, _) => Vec3.Zero,
        uSteps = 1,
        vSteps = 1,
        videoTexture = Some(videoTexture)
      )
    )

    objects.map(_.toObjectSpec.videoTexture) should contain only Some(videoTexture)

  // No object type is command-line only: the platonic solids, Cone, menger4d and
  // hexadecachoron4d had no DSL class until 2026-10. A new CLI type fails here until it has one.
  "The DSL" should "have a class for every command-line object type" in:
    val written = List[SceneObject](
      Sphere(), Cube(), Tetrahedron(), Octahedron(), Dodecahedron(), Icosahedron(), Cone(),
      Cylinder(), Sponge(VolumeFilling, 1f), Sponge(SurfaceUnfolding, 1f), Sponge(CubeSponge, 1f),
      Sponge(RecursiveIAS, 1f), Tesseract(), Pentachoron(), Hexadecachoron(),
      Icositetrachoron(), Hecatonicosachoron(), Hexacosichoron(),
      TesseractSponge(VolumeRemoving, 1f), TesseractSponge(SurfaceSubdividing, 1f),
      Menger4D(level = 1f), PentachoronGasket(level = 1f), HexadecachoronGasket(level = 1f),
      ParametricSurface(f = (_, _) => Vec3.Zero, uSteps = 1, vSteps = 1),
      Curve(points = Seq(Vec3(0f, 0f, 0f), Vec3(1f, 0f, 0f), Vec3(1f, 1f, 0f), Vec3(0f, 1f, 0f))),
      LSystem(axiom = "F", rules = Map('F' -> "F"))
    ).map(_.toObjectSpec.objectType).toSet
    // A plane is a Scene's `planes` entry, not a SceneObject.
    ObjectType.VALID_TYPES -- written shouldBe Set("plane")

  "The platonic solids" should "carry edge tubes" in:
    val tube = Some(0.02f)
    List(Tetrahedron(edgeRadius = tube), Octahedron(edgeRadius = tube),
      Dodecahedron(edgeRadius = tube), Icosahedron(edgeRadius = tube)
    ).map(_.toObjectSpec.edgeRadius) should contain only tube

  "Cone.toObjectSpec" should "carry apex, base and radius" in:
    val spec = Cone(apex = Some(Vec3(0f, 2f, 0f)), base = Some(Vec3(0f, -1f, 0f)),
      radius = Some(0.5f)).toObjectSpec
    spec.apex shouldBe Some((0f, 2f, 0f))
    spec.base shouldBe Some((0f, -1f, 0f))
    spec.radius shouldBe Some(0.5f)

  // menger#156: a round plinth for the Sprint 38 scene.
  "Cylinder" should "default to pos, size and no end points" in:
    val spec = Cylinder(material = Some(Material.Matte)).toObjectSpec
    spec.objectType shouldBe "cylinder"
    (spec.x, spec.y, spec.z, spec.size) shouldBe (0f, 0f, 0f, 1f)
    (spec.p0, spec.p1, spec.cylinderRadius) shouldBe (None, None, None)

  "Cylinder.toObjectSpec" should "carry p0, p1 and radius" in:
    val spec = Cylinder(p0 = Some(Vec3(0f, -0.1f, 0f)), p1 = Some(Vec3(0f, 0f, 0f)),
      radius = Some(1.5f)).toObjectSpec
    spec.p0 shouldBe Some((0f, -0.1f, 0f))
    spec.p1 shouldBe Some((0f, 0f, 0f))
    spec.cylinderRadius shouldBe Some(1.5f)

  it should "leave the cone's geometry empty" in:
    Cylinder(radius = Some(0.3f)).toObjectSpec.radius shouldBe None

  "Cylinder" should "reject a radius that is not positive and finite, naming the field" in:
    for r <- List(0f, -1f, Float.NaN, Float.PositiveInfinity) do
      the[IllegalArgumentException] thrownBy Cylinder(radius = Some(r)) should
        have message s"requirement failed: Radius must be positive and finite, got $r"

  it should "reject a non-positive size, naming the field" in:
    the[IllegalArgumentException] thrownBy Cylinder(size = 0f) should
      have message "requirement failed: Size must be positive, got 0.0"

  it should "reject equal end points, naming them" in:
    val p = Some(Vec3(1f, 2f, 3f))
    val e = the[IllegalArgumentException] thrownBy Cylinder(p0 = p, p1 = p)
    e.getMessage should include("p0 and p1 must differ")

  // A lone end point could land on the derived other one: p0 = (0, 0.5, 0) is the default p1.
  it should "reject a single end point" in:
    for (p0, p1) <- List((Some(Vec3(0f, 0.5f, 0f)), None), (None, Some(Vec3(0f, -0.5f, 0f)))) do
      the[IllegalArgumentException] thrownBy Cylinder(p0 = p0, p1 = p1) should
        have message "requirement failed: p0 and p1 go together: give both or neither"

  it should "reject non-finite end points" in:
    val e = the[IllegalArgumentException] thrownBy
      Cylinder(p0 = Some(Vec3(0f, Float.NaN, 0f)), p1 = Some(Vec3(0f, 1f, 0f)))
    e.getMessage should include("p0 and p1 must be finite")

  "Sierpinski4D" should "still build a PentachoronGasket under its old name" in:
    @annotation.nowarn("cat=deprecation")
    val old = Sierpinski4D(level = 2f)
    old shouldBe PentachoronGasket(level = 2f)
    old.toObjectSpec.objectType shouldBe "sierpinski4d"

  "TesseractSpongeType" should "have correct type names" in:
    TesseractSpongeType.VolumeRemoving.objectTypeName shouldBe "tesseract-sponge-volume"
    TesseractSpongeType.SurfaceSubdividing.objectTypeName shouldBe "tesseract-sponge-surface"

  "Tesseract" should "be constructible with defaults" in:
    val tesseract = Tesseract()
    tesseract.pos shouldBe Vec3.Zero
    tesseract.size shouldBe 1.0f
    tesseract.material shouldBe None
    tesseract.color shouldBe None
    tesseract.projection shouldBe None

  it should "be constructible with material only" in:
    val tesseract = Tesseract(Material.Glass)
    tesseract.pos shouldBe Vec3.Zero
    tesseract.material shouldBe Some(Material.Glass)

  it should "be constructible with position and material" in:
    val tesseract = Tesseract(Vec3(1f, 2f, 3f), Material.Gold)
    tesseract.pos shouldBe Vec3(1f, 2f, 3f)
    tesseract.material shouldBe Some(Material.Gold)

  it should "be constructible with position, material, and size" in:
    val tesseract = Tesseract(Vec3(1f, 2f, 3f), Material.Chrome, 0.5f)
    tesseract.size shouldBe 0.5f

  it should "accept projection parameters" in:
    import menger.spec.Projection4DSpec
    val projection = Projection4DSpec(eyeW = 3.0f, screenW = 1.5f)
    val tesseract = Tesseract(projection = Some(projection))
    tesseract.projection shouldBe Some(projection)

  it should "accept edge parameters" in:
    val tesseract = Tesseract(edgeRadius = Some(0.02f), edgeMaterial = Some(Material.Chrome))
    tesseract.edgeRadius shouldBe Some(0.02f)
    tesseract.edgeMaterial shouldBe Some(Material.Chrome)

  it should "validate positive size" in:
    an[IllegalArgumentException] should be thrownBy Tesseract(size = 0f)
    an[IllegalArgumentException] should be thrownBy Tesseract(size = -1f)

  it should "validate non-negative IOR" in:
    an[IllegalArgumentException] should be thrownBy Tesseract(ior = -0.1f)

  "Tesseract.toObjectSpec" should "create correct ObjectSpec" in:
    val tesseract = Tesseract(Vec3(1f, 2f, 3f), Material.Glass, 0.5f)
    val spec = tesseract.toObjectSpec

    spec.objectType shouldBe "tesseract"
    spec.x shouldBe 1f
    spec.y shouldBe 2f
    spec.z shouldBe 3f
    spec.size shouldBe 0.5f
    spec.level shouldBe None
    spec.material shouldBe Some(Material.Glass.toCoreMaterial)

  it should "include projection when specified" in:
    import menger.spec.Projection4DSpec
    val projection = Projection4DSpec(eyeW = 3.0f, screenW = 1.5f)
    val tesseract = Tesseract(projection = Some(projection))
    val spec = tesseract.toObjectSpec
    spec.projection4D shouldBe Some(projection)

  it should "include edge parameters when specified" in:
    val tesseract = Tesseract(edgeRadius = Some(0.02f), edgeMaterial = Some(Material.Chrome))
    val spec = tesseract.toObjectSpec
    spec.edgeRadius shouldBe Some(0.02f)
    spec.edgeMaterial shouldBe Some(Material.Chrome.toCoreMaterial)

  "TesseractSponge" should "be constructible with type and level" in:
    val sponge = TesseractSponge(VolumeRemoving, level = 1f)
    sponge.spongeType shouldBe VolumeRemoving
    sponge.level shouldBe 1f
    sponge.pos shouldBe Vec3.Zero

  it should "be constructible with type, level, and material" in:
    val sponge = TesseractSponge(SurfaceSubdividing, level = 2f, Material.Glass)
    sponge.material shouldBe Some(Material.Glass)

  it should "be constructible with type, level, material, and size" in:
    val sponge = TesseractSponge(VolumeRemoving, level = 1f, Material.Glass, size = 2.0f)
    sponge.size shouldBe 2.0f

  it should "be constructible with position" in:
    val sponge = TesseractSponge(Vec3(4f, 0f, 0f), SurfaceSubdividing, level = 1f)
    sponge.pos shouldBe Vec3(4f, 0f, 0f)

  it should "be constructible with position and material" in:
    val sponge = TesseractSponge(Vec3(4f, 0f, 0f), VolumeRemoving, level = 1f, Material.Glass)
    sponge.pos shouldBe Vec3(4f, 0f, 0f)
    sponge.material shouldBe Some(Material.Glass)

  it should "accept projection parameters" in:
    import menger.spec.Projection4DSpec
    val projection = Projection4DSpec(eyeW = 3.0f, screenW = 1.5f)
    val sponge = TesseractSponge(VolumeRemoving, level = 1f, projection = Some(projection))
    sponge.projection shouldBe Some(projection)

  it should "validate non-negative level" in:
    an[IllegalArgumentException] should be thrownBy TesseractSponge(VolumeRemoving, level = -1f)

  // Usability review 2026-09 (T1#3): the DSL had no upper bound at all -- only the CLI path did.
  it should "reject a level above the hard maximum" in:
    an[IllegalArgumentException] should be thrownBy
      TesseractSponge(VolumeRemoving, level = (ResourceLimits.tesseractSpongeVolumeLevel.max + 1).toFloat)

  it should "accept a level at the hard maximum" in:
    noException should be thrownBy
      TesseractSponge(VolumeRemoving, level = ResourceLimits.tesseractSpongeVolumeLevel.max.toFloat)

  it should "validate positive size" in:
    an[IllegalArgumentException] should be thrownBy TesseractSponge(VolumeRemoving, level = 1f, size = 0f)

  "TesseractSponge.toObjectSpec" should "create correct ObjectSpec with VolumeRemoving type" in:
    val sponge = TesseractSponge(Vec3(4f, 0f, 0f), VolumeRemoving, level = 1f, Material.Glass, size = 2.0f)
    val spec = sponge.toObjectSpec

    spec.objectType shouldBe "tesseract-sponge-volume"
    spec.x shouldBe 4f
    spec.y shouldBe 0f
    spec.z shouldBe 0f
    spec.size shouldBe 2.0f
    spec.level shouldBe Some(1f)
    spec.material shouldBe Some(Material.Glass.toCoreMaterial)

  it should "create correct ObjectSpec with SurfaceSubdividing type" in:
    val sponge = TesseractSponge(SurfaceSubdividing, level = 2f)
    val spec = sponge.toObjectSpec
    spec.objectType shouldBe "tesseract-sponge-surface"

  it should "include projection when specified" in:
    import menger.spec.Projection4DSpec
    val projection = Projection4DSpec(eyeW = 3.0f, screenW = 1.5f)
    val sponge = TesseractSponge(VolumeRemoving, level = 1f, projection = Some(projection))
    val spec = sponge.toObjectSpec
    spec.projection4D shouldBe Some(projection)

  // Task 20.10: DSL texture syntax
  "Sphere" should "support normalMap in DSL" in:
    val sphere = Sphere(normalMap = Some("wall_n.png"))
    sphere.toObjectSpec.normalMap shouldBe Some("wall_n.png")

  it should "support roughnessMap in DSL" in:
    val sphere = Sphere(roughnessMap = Some("wall_r.png"))
    sphere.toObjectSpec.roughnessMap shouldBe Some("wall_r.png")

  it should "support proceduralType in DSL" in:
    val sphere = Sphere(proceduralType = 5)
    sphere.toObjectSpec.proceduralType shouldBe 5

  it should "support proceduralScale in DSL" in:
    val sphere = Sphere(proceduralScale = 3.0f)
    sphere.toObjectSpec.proceduralScale shouldBe 3.0f

  it should "default normalMap and roughnessMap to None" in:
    val sphere = Sphere()
    sphere.toObjectSpec.normalMap shouldBe None
    sphere.toObjectSpec.roughnessMap shouldBe None

  it should "default proceduralType to 0 and proceduralScale to 1.0" in:
    val sphere = Sphere()
    sphere.toObjectSpec.proceduralType shouldBe 0
    sphere.toObjectSpec.proceduralScale shouldBe 1.0f

  "Cube" should "support normalMap in DSL" in:
    val cube = Cube(normalMap = Some("brick_n.png"))
    cube.toObjectSpec.normalMap shouldBe Some("brick_n.png")

  "Sponge" should "support normalMap in DSL" in:
    val sponge = Sponge(VolumeFilling, level = 2f, normalMap = Some("rock_n.png"))
    sponge.toObjectSpec.normalMap shouldBe Some("rock_n.png")

  it should "support proceduralType in DSL" in:
    val sponge = Sponge(VolumeFilling, level = 2f, proceduralType = 8)
    sponge.toObjectSpec.proceduralType shouldBe 8

  // Usability review 2026-09 (T1#3): LSystem already capped iterations at construction time;
  // these pin that behavior to ResourceLimits' single source instead of a local magic number.
  "LSystem" should "accept iterations at the hard maximum" in:
    noException should be thrownBy
      LSystem("F", Map('F' -> "F+F"), iterations = ResourceLimits.lsystemMaxIterations)

  it should "reject iterations above the hard maximum" in:
    an[IllegalArgumentException] should be thrownBy
      LSystem("F", Map('F' -> "F+F"), iterations = ResourceLimits.lsystemMaxIterations + 1)

  // Usability review 2026-09 (T1#3): was a log-and-continue warning, now a hard reject.
  "ParametricSurface" should "reject a uSteps * vSteps product above the hard maximum" in:
    val tooManySamples = (ResourceLimits.parametricSurfaceMaxSamples + 1).toInt
    an[IllegalArgumentException] should be thrownBy
      ParametricSurface(f = (_, _) => Vec3.Zero, uSteps = tooManySamples, vSteps = 1)

  it should "accept a uSteps * vSteps product at the hard maximum" in:
    noException should be thrownBy
      ParametricSurface(
        f = (_, _) => Vec3.Zero,
        uSteps = ResourceLimits.parametricSurfaceMaxSamples.toInt,
        vSteps = 1
      )
