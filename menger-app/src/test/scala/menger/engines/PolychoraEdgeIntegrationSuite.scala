package menger.engines

import menger.ObjectSpec
import menger.common.ObjectType
import menger.engines.scene.TesseractEdgeSceneBuilder
import menger.objects.higher_d.Hexadecachoron
import menger.objects.higher_d.Icositetrachoron
import menger.objects.higher_d.Mesh4DProjection
import menger.objects.higher_d.Pentachoron
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Integration tests for edge rendering on 4D polychora (pentachoron, 16-cell, 24-cell).
  *
  * Covers the bug fixed in TesseractEdgeSceneBuilder where edge-material on any
  * polychora type crashed with "Unknown 4D object type". */
class PolychoraEdgeIntegrationSuite extends AnyFlatSpec with Matchers:

  // === ObjectSpec parsing — edge rendering on polychora ===

  "Pentachoron edge rendering" should "parse with edge-material" in:
    val result = ObjectSpec.parse("type=pentachoron:size=0.8:material=film:edge-material=gold:edge-radius=0.02")
    result shouldBe a[Right[?, ?]]
    val spec = result.toOption.get
    spec.objectType shouldBe "pentachoron"
    spec.hasEdgeRendering shouldBe true
    spec.edgeMaterial shouldBe defined
    spec.edgeRadius shouldBe Some(0.02f)

  it should "parse with edge-radius only" in:
    val result = ObjectSpec.parse("type=pentachoron:edge-radius=0.03")
    result shouldBe a[Right[?, ?]]
    val spec = result.toOption.get
    spec.hasEdgeRendering shouldBe true

  it should "parse with emissive edges" in:
    val result = ObjectSpec.parse("type=pentachoron:edge-color=#ff00ff:edge-emission=3.0:edge-radius=0.025")
    result shouldBe a[Right[?, ?]]
    val spec = result.toOption.get
    spec.edgeMaterial shouldBe defined
    spec.hasEdgeRendering shouldBe true

  "16-cell edge rendering" should "parse with edge-material" in:
    val result = ObjectSpec.parse("type=16-cell:size=0.8:material=film:edge-material=chrome:edge-radius=0.02")
    result shouldBe a[Right[?, ?]]
    val spec = result.toOption.get
    spec.objectType shouldBe "16-cell"
    spec.hasEdgeRendering shouldBe true
    spec.edgeMaterial shouldBe defined
    spec.edgeRadius shouldBe Some(0.02f)

  it should "parse with edge-radius only" in:
    val result = ObjectSpec.parse("type=16-cell:edge-radius=0.015")
    result shouldBe a[Right[?, ?]]
    result.toOption.get.hasEdgeRendering shouldBe true

  "24-cell edge rendering" should "parse with edge-material" in:
    val result = ObjectSpec.parse("type=24-cell:size=0.8:material=film:edge-material=gold:edge-radius=0.02")
    result shouldBe a[Right[?, ?]]
    val spec = result.toOption.get
    spec.objectType shouldBe "24-cell"
    spec.hasEdgeRendering shouldBe true
    spec.edgeMaterial shouldBe defined
    spec.edgeRadius shouldBe Some(0.02f)

  it should "parse with all parameters" in:
    val result = ObjectSpec.parse(
      "type=24-cell:size=0.8:material=film:edge-material=gold:edge-radius=0.02" +
      ":rot-xw=30:rot-yw=20:eye-w=4.0:screen-w=1.5"
    )
    result shouldBe a[Right[?, ?]]
    val spec = result.toOption.get
    spec.hasEdgeRendering shouldBe true
    val proj = spec.projection4D.get
    proj.rotXW shouldBe 30f
    proj.eyeW shouldBe 4.0f

  // === TesseractEdgeSceneBuilder.calculateRequiredInstances ===
  // Verifies no crash and correct edge counts for triangular polychora faces.

  given menger.ProfilingConfig = menger.ProfilingConfig.disabled

  "TesseractEdgeSceneBuilder.calculateRequiredInstances" should "handle pentachoron with edges" in:
    val spec = ObjectSpec.parse("type=pentachoron:size=0.8:edge-material=gold:edge-radius=0.02")
      .toOption.get
    val builder = TesseractEdgeSceneBuilder(".")
    // 1 face mesh + 10 edges = 11
    builder.calculateRequiredInstances(List(spec)) shouldBe 11

  it should "handle 16-cell with edges" in:
    val spec = ObjectSpec.parse("type=16-cell:size=0.8:edge-material=chrome:edge-radius=0.02")
      .toOption.get
    val builder = TesseractEdgeSceneBuilder(".")
    // 1 face mesh + 24 edges = 25
    builder.calculateRequiredInstances(List(spec)) shouldBe 25

  it should "handle 24-cell with edges" in:
    val spec = ObjectSpec.parse("type=24-cell:size=0.8:edge-material=gold:edge-radius=0.02")
      .toOption.get
    val builder = TesseractEdgeSceneBuilder(".")
    // 1 face mesh + 96 edges = 97
    builder.calculateRequiredInstances(List(spec)) shouldBe 97

  // === Mesh projection sanity check — triangular faces project without crash ===

  "Pentachoron mesh projection" should "produce valid triangle mesh with default params" in:
    val data = Mesh4DProjection(Pentachoron(size = 0.8f)).toTriangleMesh
    data.numVertices shouldBe 30
    data.numTriangles shouldBe 10
    data.vertices.foreach(v => v.isNaN shouldBe false)

  "16-cell mesh projection" should "produce valid triangle mesh with default params" in:
    val data = Mesh4DProjection(Hexadecachoron(size = 0.8f)).toTriangleMesh
    data.numVertices shouldBe 96
    data.numTriangles shouldBe 32
    data.vertices.foreach(v => v.isNaN shouldBe false)

  "24-cell mesh projection" should "produce valid triangle mesh with default params" in:
    val data = Mesh4DProjection(Icositetrachoron(size = 0.8f)).toTriangleMesh
    data.numVertices shouldBe 288
    data.numTriangles shouldBe 96
    data.vertices.foreach(v => v.isNaN shouldBe false)

  // === Type classification ===

  "ObjectType" should "classify all polychora as projected 4D" in:
    ObjectType.isProjected4D("pentachoron") shouldBe true
    ObjectType.isProjected4D("16-cell") shouldBe true
    ObjectType.isProjected4D("24-cell") shouldBe true

  it should "classify all polychora as valid types" in:
    ObjectType.isValid("pentachoron") shouldBe true
    ObjectType.isValid("16-cell") shouldBe true
    ObjectType.isValid("24-cell") shouldBe true
