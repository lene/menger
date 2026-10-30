package menger.tools

import java.io.File
import java.nio.file.Files

import menger.dsl.FieldConstraints
import menger.dsl.ProceduralPresets
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import upickle.default.write

/** The manifest as the single source the agent derives its vocabulary from (review 2026-10-14,
  * menger#81, #83, #114, msa#28): each section comes from the code that owns it, and its JSON
  * shape changes only with a schemaVersion bump. */
class ManifestContentSuite extends AnyFlatSpec with Matchers:

  private val manifest = ManifestGenerator.buildManifest()

  "proceduralPresets" should "be exactly optix-jni's ProceduralType ids" in:
    val optixIds = io.github.lene.optix.ProceduralType.getClass.getDeclaredMethods.toList
      .filter(m => m.getParameterCount == 0 && m.getReturnType == classOf[Int])
      .map(_.invoke(io.github.lene.optix.ProceduralType))
      .collect { case id: java.lang.Integer => id.intValue }
      .toSet
    manifest.proceduralPresets.map(_.id).toSet shouldBe optixIds
    manifest.proceduralPresets.map(_.name) shouldBe ProceduralPresets.All.map(_.name)

  // The agent's hand list had the first six; Parchment (40% opaque) lets light through too.
  "materials" should "mark the presets that let light through" in:
    manifest.materials.filter(_.transparent).map(_.name).toSet shouldBe
      Set("Glass", "Water", "Diamond", "GlassDispersive", "DiamondDispersive", "Film", "Parchment")

  "traits" should "mark holed objects and positioned lights" in:
    manifest.objects.filter(_.traits.contains("holed")).map(_.name).toSet shouldBe
      DslSemantics.holedTypes
    manifest.lights.filter(_.traits.contains("positioned")).map(_.name).toSet shouldBe
      Set("Point", "AreaLight")

  "every bounded field" should "carry its bounds in the manifest" in:
    val types = (manifest.objects ++ manifest.lights ++ manifest.sceneComposition :+
      manifest.renderSettings).map(t => t.name -> t).toMap
    FieldConstraints.fields.foreach { case ((typeName, field), bounds) =>
      withClue(s"$typeName.$field") {
        val f = types.get(typeName).flatMap(_.fields.find(_.name == field))
        f shouldBe defined
        (f.get.min, f.get.max) shouldBe (bounds.min, bounds.max)
      }
    }
    manifest.productCaps.map(c => (c.`type`, c.fields)) shouldBe
      List(("ParametricSurface", List("uSteps", "vSteps")))

  // menger#95: the unit of an angle used to be a prose convention; it is read off the field's
  // type now (FieldUnits), so a field retyped Degrees <-> Radians changes it here.
  "angle-valued fields" should "state their unit" in:
    val types = (manifest.objects ++ manifest.sceneComposition).map(t => t.name -> t).toMap
    def unit(typeName: String, field: String): Option[String] =
      types(typeName).fields.find(_.name == field).flatMap(_.unit)
    unit("Projection4DSpec", "rotXW") shouldBe Some("degrees")
    unit("Projection4DSpec", "rotYW") shouldBe Some("degrees")
    unit("Projection4DSpec", "rotZW") shouldBe Some("degrees")
    unit("LSystem", "angleDegrees") shouldBe Some("degrees")
    unit("Camera", "fov") shouldBe Some("degrees")
    unit("Sphere", "rotation") shouldBe Some("radians")
    unit("Transform", "rotation") shouldBe Some("radians")

  // The fitness function for menger#95: a field named like an angle (rot..., ...Angle..., fov)
  // that is a raw Float has no unit, so a new one fails here until it is typed Degrees/Radians.
  it should "include every field whose name says it is an angle" in:
    val angleNamed = """(?i).*(rot|angle|fov).*""".r
    val unitless = (manifest.objects ++ manifest.sceneComposition).flatMap { t =>
      t.fields.collect { case f if angleNamed.matches(f.name) && f.unit.isEmpty =>
        s"${t.name}.${f.name}"
      }
    }
    unitless shouldBe empty

  it should "leave other numeric fields without a unit" in:
    val types = (manifest.objects ++ manifest.sceneComposition).map(t => t.name -> t).toMap
    types("Projection4DSpec").fields.find(_.name == "eyeW").flatMap(_.unit) shouldBe None
    types("Sphere").fields.find(_.name == "size").flatMap(_.unit) shouldBe None

  "compileSurface" should "list packages only, including menger.spec" in:
    manifest.compileSurface.packages should contain allOf ("menger.dsl", "menger.spec")
    manifest.compileSurface.packages shouldBe sorted

  "Caustics" should "call its radius-reduction factor radiusReduction, not alpha" in:
    val fields = manifest.sceneComposition.find(_.name == "Caustics").get.fields.map(_.name)
    fields should contain ("radiusReduction")
    fields should not contain "alpha"

  "contentRevision" should "ignore version pins and follow content" in:
    ManifestGenerator.contentRevisionOf(manifest.copy(optixJniVersion = "9.9.9")) shouldBe
      manifest.contentRevision
    ManifestGenerator.contentRevisionOf(manifest.copy(conventions = Nil)) should not be
      manifest.contentRevision

  // The manifest's JSON shape (every key path) is pinned with the schemaVersion that describes
  // it, so a shape change without a version bump fails here. After a deliberate change: bump
  // SchemaVersion, then UPDATE_MANIFEST_SHAPE=1 sbt "mengerApp/testOnly *ManifestContentSuite".
  "schemaVersion" should "change whenever the manifest's shape changes" in:
    val shapeFile = new File(sys.props("user.dir"), "src/test/resources/manifest-shape.txt")
    val current =
      (ManifestGenerator.SchemaVersion :: keyPathsSorted(ujson.read(write(manifest)))).mkString("\n")
    if sys.env.contains("UPDATE_MANIFEST_SHAPE") then Files.writeString(shapeFile.toPath, current + "\n")
    val pinned = Files.readString(shapeFile.toPath).trim
    val pinnedVersion = pinned.linesIterator.next()
    withClue(s"manifest shape pinned for $pinnedVersion differs; bump SchemaVersion and re-pin: ") {
      current shouldBe pinned
    }

  private def keyPaths(value: ujson.Value, prefix: String = ""): List[String] = value match
    case o: ujson.Obj => o.value.toList.flatMap((k, v) => (prefix + k) :: keyPaths(v, prefix + k + "."))
    case a: ujson.Arr => a.value.toList.flatMap(keyPaths(_, prefix)).distinct
    case _            => Nil
  private def keyPathsSorted(value: ujson.Value): List[String] = keyPaths(value).distinct.sorted
