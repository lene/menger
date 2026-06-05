package menger.engines.scene

import menger.ObjectSpec
import menger.Projection4DSpec
import menger.common.ProfilingConfig
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MeshFactorySpec extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  // --- 4D types → Gpu4D plan ---

  "MeshFactory.createUpload" should "return Gpu4D plan for tesseract" in:
    val plan = MeshFactory.createUpload(ObjectSpec("tesseract"))
    plan shouldBe a[MeshUploadPlan.Gpu4D]

  it should "return Gpu4D plan for pentachoron" in:
    val plan = MeshFactory.createUpload(ObjectSpec("pentachoron"))
    plan shouldBe a[MeshUploadPlan.Gpu4D]

  it should "return Gpu4D plan for 16-cell" in:
    val plan = MeshFactory.createUpload(ObjectSpec("16-cell"))
    plan shouldBe a[MeshUploadPlan.Gpu4D]

  it should "return Gpu4D plan for 24-cell" in:
    val plan = MeshFactory.createUpload(ObjectSpec("24-cell"))
    plan shouldBe a[MeshUploadPlan.Gpu4D]

  it should "return Gpu4D plan for tesseract-sponge with level" in:
    val plan = MeshFactory.createUpload(ObjectSpec("tesseract-sponge", level = Some(1f)))
    plan shouldBe a[MeshUploadPlan.Gpu4D]

  it should "return Gpu4D plan for tesseract-sponge-2 with level" in:
    val plan = MeshFactory.createUpload(ObjectSpec("tesseract-sponge-2", level = Some(1f)))
    plan shouldBe a[MeshUploadPlan.Gpu4D]

  // --- buffer contents ---

  it should "produce non-empty quad buffer for tesseract" in:
    MeshFactory.createUpload(ObjectSpec("tesseract")) match
      case MeshUploadPlan.Gpu4D(quads, vpf, _) =>
        quads.length should be > 0
        vpf should be > 0
      case other => fail(s"expected Gpu4D, got $other")

  it should "produce 384-float buffer for unit tesseract (24 faces × 4 verts × 4 components)" in:
    MeshFactory.createUpload(ObjectSpec("tesseract")) match
      case MeshUploadPlan.Gpu4D(quads, vpf, _) =>
        quads.length shouldBe 384
        vpf shouldBe 4
      case other => fail(s"expected Gpu4D, got $other")

  // --- projection params ---

  it should "use default projection when spec has none" in:
    MeshFactory.createUpload(ObjectSpec("tesseract")) match
      case MeshUploadPlan.Gpu4D(_, _, proj) => proj shouldBe Projection4DSpec.default
      case other => fail(s"expected Gpu4D, got $other")

  it should "use spec projection when provided" in:
    val customProj = Projection4DSpec(eyeW = 4.0f, screenW = 2.0f, rotXW = 15f, rotYW = 10f, rotZW = 5f)
    MeshFactory.createUpload(ObjectSpec("tesseract", projection4D = Some(customProj))) match
      case MeshUploadPlan.Gpu4D(_, _, proj) => proj shouldBe customProj
      case other => fail(s"expected Gpu4D, got $other")

  // --- 3D types → Cpu plan ---

  it should "return Cpu plan for cube" in:
    val plan = MeshFactory.createUpload(ObjectSpec("cube"))
    plan shouldBe a[MeshUploadPlan.Cpu]

  it should "return Cpu plan for sponge-volume" in:
    val plan = MeshFactory.createUpload(ObjectSpec("sponge-volume", level = Some(1f)))
    plan shouldBe a[MeshUploadPlan.Cpu]

  it should "return Cpu plan for tetrahedron" in:
    val plan = MeshFactory.createUpload(ObjectSpec("tetrahedron"))
    plan shouldBe a[MeshUploadPlan.Cpu]
