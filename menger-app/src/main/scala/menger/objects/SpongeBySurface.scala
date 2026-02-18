package menger.objects

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.Model
import com.badlogic.gdx.graphics.g3d.ModelInstance
import com.badlogic.gdx.graphics.g3d.utils.MeshBuilder
import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData
import menger.common.float2string
import menger.objects.Direction.Z

// Menger Sponge Surface-Based Generator
//
// Generates a Menger sponge by recursive subdivision of its 6 outer faces.
// Unlike volume-based approaches that start with a cube and carve holes,
// this builds only the visible surfaces, which is more memory-efficient.
//
// Algorithm:
//   1. Start with 6 faces (one per cube side)
//   2. Each face subdivides into 12 sub-faces per level (see Face.scala)
//   3. After N levels, render all accumulated faces as a mesh
//
// The surfaces() method iterates from level N down to 0, applying
// subdivision at each step. This reverse iteration allows the same
// Face.subdivide() logic to be applied uniformly.
//
// Supports fractional levels by blending between integer levels
// (nextLevelSponge + transparentSponge for smooth LOD transitions).

class SpongeBySurface(
  val center: Vector3 = Vector3.Zero, val scale: Float = 1f,
  val level: Float, val material: Material = Builder.WHITE_MATERIAL, val primitiveType: Int = GL20.GL_TRIANGLES
)(using val profilingConfig: menger.ProfilingConfig) extends Geometry(center, scale) with FractionalLevelSponge with TriangleMeshSource:
  require(level >= 0, "Level must be non-negative")

  override protected def createInstance(
    center: Vector3, scale: Float, level: Float, material: Material, primitiveType: Int
  ): Geometry & FractionalLevelSponge =
    SpongeBySurface(center, scale, level, material, primitiveType)

  override def getModel: List[ModelInstance] = logTime("getModel") {
    if level.isValidInt then getIntegerModel
    else List(
      nextLevelSponge.map(_.getModel).getOrElse(Nil),
      transparentSponge.map(_.getModel).getOrElse(Nil)
    ).flatten
  }

  private lazy val getIntegerModel =
    val facingPlusX = transformed(ModelInstance(mesh), scale, center, 0, 1, 0, 90)
    val facingMinusX = transformed(ModelInstance(mesh), scale, center, 0, 1, 0, -90)
    val facingPlusY = transformed(ModelInstance(mesh), scale, center, 1, 0, 0, 90)
    val facingMinusY = transformed(ModelInstance(mesh), scale, center, 1, 0, 0, -90)
    val facingPlusZ = transformed(ModelInstance(mesh), scale, center, 0, 1, 0, 0)
    val facingMinusZ = transformed(ModelInstance(mesh), scale, center, 0, 1, 0, 180)

    List(facingPlusX, facingMinusX, facingPlusY, facingMinusY, facingPlusZ, facingMinusZ)

  private def transformed(
    modelInstance: ModelInstance, scale: Float, xlate: Vector3, axisX: Float, axisY: Float, axisZ: Float, angle: Float
  ): ModelInstance =
    modelInstance.transform.translate(xlate)
    modelInstance.transform.rotate(axisX, axisY, axisZ, angle)
    modelInstance.transform.translate(0, 0, scale / 2)
    modelInstance.transform.scale(scale, scale, scale)
    modelInstance

  override def toString: String = s"SpongeBySurface(level=${float2string(level)}, ${6 * faces.size} faces)"

  // Apply subdivision N times (from level down to 1), accumulating sub-faces
  private[objects] def surfaces(startFace: Face): Seq[Face] =
    val faces = Seq(startFace)
    level.toInt.until(0, -1).foldLeft(faces)(
      (faces, _) => faces.flatMap(_.subdivide())
    )

  lazy val faces: Seq[Face] = logTime("faces") { surfaces(Face(0, 0, 0, 1, Z)) }

  lazy val mesh: Model = logTime("mesh") {
      Builder.modelBuilder.begin()
      faces.grouped(MeshBuilder.MAX_VERTICES / 4).foreach(facesPart =>
        val meshBuilder = Builder.modelBuilder.part("sponge", primitiveType, Builder.DEFAULT_FLAGS, material)
        facesPart.foreach(face => meshBuilder.rect.tupled(face.vertices))
      )
      Builder.modelBuilder.end()
    }

  // Generate triangle mesh for all 6 cube faces
  // Each face is offset by half the scale in its normal direction (on the cube surface)
  override def toTriangleMesh: TriangleMeshData = logTime("toTriangleMesh") {
    if level.isValidInt then getIntegerMesh
    else getFractionalMesh
  }

  private def getFractionalMesh: TriangleMeshData =
    val fractionalPart = level - level.floor
    val alphaTransparent = 1.0f - fractionalPart

    // Generate both level geometries.
    // Expand skin faces outward along normals to prevent z-fighting (same fix as
    // SpongeByVolume.getFractionalMesh — see FractionalLevelSponge for rationale).
    val nextLevel = SpongeBySurface(center, scale, (level + 1).floor, material, primitiveType).toTriangleMesh
    val currentLevel = TriangleMeshData.expandAlongNormals(
      SpongeBySurface(center, scale, level.floor, material, primitiveType).toTriangleMesh,
      FractionalLevelSponge.SkinNormalOffset
    )

    // Assign per-vertex alpha: next level opaque, current level transparent
    val nextWithAlpha = TriangleMeshData.withAlpha(nextLevel, 1.0f)
    val currentWithAlpha = TriangleMeshData.withAlpha(currentLevel, alphaTransparent)

    // Merge into single mesh
    TriangleMeshData.merge(Seq(nextWithAlpha, currentWithAlpha))

  private def getIntegerMesh: TriangleMeshData =
    val half = scale / 2
    // Create initial faces offset by half in their normal direction (on the cube surface)
    val allFaces = Direction.values.flatMap { dir =>
      val offset = half * dir.sign
      val (fx, fy, fz) = dir match
        case Direction.X | Direction.negX => (center.x + offset, center.y, center.z)
        case Direction.Y | Direction.negY => (center.x, center.y + offset, center.z)
        case Direction.Z | Direction.negZ => (center.x, center.y, center.z + offset)
      surfaces(Face(fx, fy, fz, scale, dir))
    }
    val meshes = allFaces.map(_.toTriangleMesh).toSeq
    TriangleMeshData.merge(meshes)
