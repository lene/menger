package menger.objects

import com.badlogic.gdx.math.Vector3
import menger.common.TriangleMeshData
import menger.common.TriangleMeshSource
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
// (see FractionalLevelSponge.buildFractionalMesh for smooth LOD transitions).

class SpongeBySurface(
  val center: Vector3 = Vector3.Zero, val scale: Float = 1f,
  val level: Float
)(using val profilingConfig: menger.common.ProfilingConfig) extends Geometry(center, scale) with FractionalLevelSponge with TriangleMeshSource:
  require(level >= 0, "Level must be non-negative")

  override def toString: String = s"SpongeBySurface(level=${float2string(level)}, ${6 * faces.size} faces)"

  // Apply subdivision N times (from level down to 1), accumulating sub-faces
  private[objects] def surfaces(startFace: Face): Seq[Face] =
    val faces = Seq(startFace)
    level.toInt.until(0, -1).foldLeft(faces)(
      (faces, _) => faces.flatMap(_.subdivide())
    )

  lazy val faces: Seq[Face] = logTime("faces") { surfaces(Face(0, 0, 0, 1, Z)) }

  // Generate triangle mesh for all 6 cube faces
  // Each face is offset by half the scale in its normal direction (on the cube surface)
  override def toTriangleMesh: TriangleMeshData = logTime("toTriangleMesh") {
    if level.isValidInt then getIntegerMesh
    else getFractionalMesh
  }

  private def getFractionalMesh: TriangleMeshData =
    buildFractionalMesh(
      nextLevelMesh    = SpongeBySurface(center, scale, (level + 1).floor).toTriangleMesh,
      currentLevelMesh = SpongeBySurface(center, scale, level.floor).toTriangleMesh
    )

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
