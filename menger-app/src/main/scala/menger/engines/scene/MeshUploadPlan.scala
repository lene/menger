package menger.engines.scene

import menger.Projection4DSpec
import menger.common.TriangleMeshData

/** Outcome of `MeshFactory.create*` — either a CPU-projected triangle mesh
  * (default path) or a 4D quad buffer that the GPU kernel will project at
  * upload time (Sprint 18.3 `--gpu-project-4d`).
  */
sealed trait MeshUploadPlan

object MeshUploadPlan:
  final case class Cpu(data: TriangleMeshData) extends MeshUploadPlan
  final case class Gpu4D(
    quads4D: Array[Float],
    vertsPerFace: Int,
    proj: Projection4DSpec
  ) extends MeshUploadPlan
