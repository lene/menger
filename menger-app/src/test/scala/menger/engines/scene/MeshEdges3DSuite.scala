package menger.engines.scene

import menger.ObjectSpec
import menger.common.ProfilingConfig
import menger.dsl.Material
import menger.dsl.Sponge
import menger.dsl.SpongeType
import menger.engines.SceneGroups
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// Usability review 2026-10, session 5 (F124, menger#132): 3D objects had no edge tubes.
class MeshEdges3DSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  private val tube = Some(0.01f)

  private def edgeCount(spec: ObjectSpec): Int = MeshEdges3D.of(MeshFactory.create(spec)).size

  "MeshEdges3D.of" should "give a cube its 12 edges, not the quads' diagonals" in:
    edgeCount(ObjectSpec(objectType = "cube", edgeRadius = tube)) shouldBe 12

  it should "drop the seams between coplanar faces of a sponge's sub-cubes" in:
    // Counted independently from the sponge's leaf cubes (a face between two kept cubes is
    // hidden; an edge stays where its faces differ in normal).
    edgeCount(ObjectSpec(objectType = "sponge-volume", level = Some(1f), edgeRadius = tube)) shouldBe 96
    edgeCount(ObjectSpec(objectType = "sponge-volume", level = Some(2f), edgeRadius = tube)) shouldBe 1296

  "MeshEdges3D.worldEdges" should "place the edges like the faces' instance" in:
    val spec = ObjectSpec(objectType = "cube", x = 3f, size = 2f, edgeRadius = tube)
    val xs = MeshEdges3D.worldEdges(spec).flatMap((p, q) => Seq(p(0), q(0)))
    xs.min shouldBe 2f +- 1e-5f
    xs.max shouldBe 4f +- 1e-5f

  "MeshEdges3D.count" should "count a fractional level's two levels, faded into each other" in:
    val spec = ObjectSpec(objectType = "sponge-volume", level = Some(1.5f), edgeRadius = tube)
    MeshEdges3D.count(spec) shouldBe 96L + 1296L

  "MeshEdges3D.appliesTo" should "take a cube or mesh sponge with edges, nothing else" in:
    MeshEdges3D.appliesTo(ObjectSpec(objectType = "cube", edgeRadius = tube)) shouldBe true
    MeshEdges3D.appliesTo(ObjectSpec(objectType = "cube")) shouldBe false
    MeshEdges3D.appliesTo(ObjectSpec(objectType = "tesseract", edgeRadius = tube)) shouldBe false
    MeshEdges3D.appliesTo(
      ObjectSpec(objectType = "sponge-recursive-ias", level = Some(2f), edgeRadius = tube)
    ) shouldBe false

  "The command line" should "accept edge parameters on a cube and refuse them on an IAS sponge" in:
    ObjectSpec.parse("type=cube:edge-radius=0.02").map(_.edgeRadius) shouldBe Right(Some(0.02f))
    ObjectSpec.parse("type=sponge-recursive-ias:level=2:edge-radius=0.02") shouldBe a[Left[?, ?]]

  "The DSL" should "carry a sponge's edges and refuse them on a recursive-IAS sponge" in:
    val sponge = Sponge(SpongeType.VolumeFilling, level = 1f, edgeRadius = tube,
      edgeMaterial = Some(Material.Gold))
    sponge.toObjectSpec.edgeRadius shouldBe tube
    sponge.toObjectSpec.edgeMaterial shouldBe defined
    an[IllegalArgumentException] should be thrownBy
      Sponge(SpongeType.RecursiveIAS, level = 2f, edgeRadius = tube)

  "SceneGroups.buildOrder" should "build 3D meshes with edges before the analytical primitives" in:
    // Their builder may reinitialize the renderer, which would discard a sphere built before.
    val cube = ObjectSpec(objectType = "cube", edgeRadius = tube)
    val sphere = ObjectSpec(objectType = "sphere")
    SceneGroups.buildOrder(List(sphere, cube)).head shouldBe List(cube)
