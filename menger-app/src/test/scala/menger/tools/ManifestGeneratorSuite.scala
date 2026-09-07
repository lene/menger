package menger.tools

import java.nio.file.Files

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import upickle.default.read

class ManifestGeneratorSuite extends AnyFlatSpec with Matchers:

  private def freshTempPath(): String =
    val dir = Files.createTempDirectory("manifest-generator-suite")
    dir.resolve("dsl-manifest.json").toString

  "ManifestGenerator.resolveOutputPath" should "default to target/dsl-manifest.json when no args given" in:
    ManifestGenerator.resolveOutputPath(Array.empty) shouldBe "target/dsl-manifest.json"

  it should "use the first argument when supplied" in:
    ManifestGenerator.resolveOutputPath(Array("some/other/path.json")) shouldBe "some/other/path.json"

  "ManifestGenerator.run" should "write a valid, versioned JSON manifest at the given path" in:
    val outputPath = freshTempPath()
    val result = ManifestGenerator.run(Array(outputPath))

    result shouldBe Right(())
    Files.exists(java.nio.file.Paths.get(outputPath)) shouldBe true

    val json = Files.readString(java.nio.file.Paths.get(outputPath))
    val manifest = read[ManifestGenerator.DslManifest](json)

    manifest.schemaVersion should not be empty
    manifest.scalaVersion shouldBe "3.8.3"
    manifest.optixJniVersion shouldBe "0.3.3"
    manifest.minDriverVersion shouldBe "580.65"

  it should "write to target/dsl-manifest.json when no output path is given" in:
    val defaultPath = java.nio.file.Paths.get("target/dsl-manifest.json")
    Files.deleteIfExists(defaultPath)
    try
      val result = ManifestGenerator.run(Array.empty)
      result shouldBe Right(())
      Files.exists(defaultPath) shouldBe true
    finally
      Files.deleteIfExists(defaultPath)

  it should "report a clear, non-empty error and not throw on an unwritable path" in:
    // A regular file cannot be treated as a directory: Files.createDirectories on a path
    // whose parent is an existing plain file fails with an IOException -- deterministic
    // regardless of OS user/permission bits (unlike chmod-based approaches, which no-op
    // for a root-run test process).
    val blockingFile = Files.createTempFile("manifest-generator-blocker", "")
    try
      val target = blockingFile.resolve("dsl-manifest.json").toString
      val result = ManifestGenerator.run(Array(target))

      result.isLeft shouldBe true
      result.left.getOrElse("") should include(target)
    finally
      Files.deleteIfExists(blockingFile)

  "The generated manifest" should "list exactly the 9 SceneObject case classes" in:
    val manifest = manifestFor(freshTempPath())
    manifest.objects.map(_.name).toSet shouldBe Set(
      "Sphere", "Cube", "Sponge", "Tesseract", "TesseractSponge",
      "Sierpinski4D", "ParametricSurface", "Curve", "LSystem"
    )
    manifest.objects should have size 9

  it should "list exactly the 12 Material presets" in:
    val manifest = manifestFor(freshTempPath())
    manifest.materials.map(_.name).toSet shouldBe Set(
      "Glass", "Water", "Diamond", "GlassDispersive", "DiamondDispersive",
      "Chrome", "Gold", "Copper", "Film", "Parchment", "Plastic", "Matte"
    )
    manifest.materials should have size 12

  it should "list exactly the 3 Light case classes" in:
    val manifest = manifestFor(freshTempPath())
    manifest.lights.map(_.name).toSet shouldBe Set("Directional", "Point", "AreaLight")

  it should "list Placement's mode functions" in:
    val manifest = manifestFor(freshTempPath())
    manifest.placement.map(_.name).toSet shouldBe Set("grid", "ring", "spiral", "scatter")

  it should "list Color's presets" in:
    val manifest = manifestFor(freshTempPath())
    manifest.colorPresets.toSet should contain allOf ("White", "Black", "Red", "Green", "Blue", "Gray")

  it should "carry Plane's constructor fields" in:
    val manifest = manifestFor(freshTempPath())
    manifest.plane.name shouldBe "Plane"
    manifest.plane.fields.map(_.name) shouldBe List("axisPosition", "color", "checkered", "material")

  it should "carry CameraPath's constructor fields" in:
    val manifest = manifestFor(freshTempPath())
    manifest.cameraPath.name shouldBe "CameraPath"
    manifest.cameraPath.fields.map(_.name) shouldBe List("positions", "lookAt", "up")

  it should "carry RenderSettings's fields and its Default/HighQuality presets" in:
    val manifest = manifestFor(freshTempPath())
    manifest.renderSettings.name shouldBe "RenderSettings"
    manifest.renderSettings.fields.map(_.name) shouldBe List(
      "shadows", "transparentShadows", "antialiasing", "aaMaxDepth",
      "aaThreshold", "maxRayDepth", "accumulation", "denoise"
    )
    manifest.renderSettingsPresets.toSet shouldBe Set("Default", "HighQuality")

  it should "list the AxisHelper axes (X, Y, Z) planes can be placed on" in:
    val manifest = manifestFor(freshTempPath())
    manifest.planeAxes.toSet shouldBe Set("X", "Y", "Z")

  it should "carry Bezier.cubic among CameraPath's helpers" in:
    val manifest = manifestFor(freshTempPath())
    val cubic = manifest.cameraPathHelpers.find(_.name == "cubic").getOrElse(
      fail("Bezier.cubic missing from generated manifest")
    )
    cubic.parameters.map(_.name) shouldBe List("p0", "p1", "p2", "p3", "t")

  it should "capture Placement.scatter's seed default, not omit it" in:
    val manifest = manifestFor(freshTempPath())
    val scatter = manifest.placement.find(_.name == "scatter").getOrElse(
      fail("Placement.scatter missing from generated manifest")
    )
    val seedParam = scatter.parameters.find(_.name == "seed").getOrElse(
      fail("Placement.scatter's seed parameter missing from generated manifest")
    )
    seedParam.default shouldBe Some("42")

  it should "capture a field's default value rather than omitting it" in:
    val manifest = manifestFor(freshTempPath())
    val sphere = manifest.objects.find(_.name == "Sphere").getOrElse(
      fail("Sphere missing from generated manifest")
    )
    val sizeField = sphere.fields.find(_.name == "size").getOrElse(
      fail("Sphere.size missing from generated manifest")
    )
    sizeField.default shouldBe defined
    sizeField.default.get should include("1.0")

  it should "leave a genuinely required field's default as None" in:
    val manifest = manifestFor(freshTempPath())
    val sponge = manifest.objects.find(_.name == "Sponge").getOrElse(
      fail("Sponge missing from generated manifest")
    )
    val levelField = sponge.fields.find(_.name == "level").getOrElse(
      fail("Sponge.level missing from generated manifest")
    )
    levelField.default shouldBe None

  private def manifestFor(outputPath: String): ManifestGenerator.DslManifest =
    ManifestGenerator.run(Array(outputPath)) shouldBe Right(())
    read[ManifestGenerator.DslManifest](Files.readString(java.nio.file.Paths.get(outputPath)))
