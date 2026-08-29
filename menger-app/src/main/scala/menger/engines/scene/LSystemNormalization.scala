package menger.engines.scene

import menger.ObjectSpec

/** Shared post-processing for L-system turtle output (Sprint 36 #16): rescales the whole
  * generated tree so its largest dimension is 1.0 and centers it on the origin. Operates on
  * already-projected `ObjectSpec`/`CurveData` points, so it's dimension-agnostic -- both
  * `LSystemTurtle3D` and `LSystemTurtle4D` call it after generating their raw output.
  * Originally lived only in `LSystemTurtle3D`; `LSystemTurtle4D` never had an equivalent,
  * which left its output at raw, un-normalized, un-centered turtle-walk scale.
  */
object LSystemNormalization:

  def normalize(specs: List[ObjectSpec]): List[ObjectSpec] =
    val allPoints = specs.flatMap { s =>
      s.curveData.map(_.points.grouped(3).map(g => (g(0), g(1), g(2))).toVector)
        .getOrElse(Vector((s.x, s.y, s.z)))
    }
    if allPoints.isEmpty then specs
    else
      val minX = allPoints.map(_._1).min
      val minY = allPoints.map(_._2).min
      val minZ = allPoints.map(_._3).min
      val maxX = allPoints.map(_._1).max
      val maxY = allPoints.map(_._2).max
      val maxZ = allPoints.map(_._3).max
      val sizeX = maxX - minX
      val sizeY = maxY - minY
      val sizeZ = maxZ - minZ
      val maxDim = math.max(math.max(sizeX, sizeY), sizeZ)
      if maxDim <= 0f then specs
      else
        val scale = 1f / maxDim
        val offsetX = (minX + maxX) / 2f
        val offsetY = (minY + maxY) / 2f
        val offsetZ = (minZ + maxZ) / 2f
        specs.map(s => normalizeSpec(s, scale, offsetX, offsetY, offsetZ))

  private def normalizeSpec(
    s: ObjectSpec, scale: Float, ox: Float, oy: Float, oz: Float
  ): ObjectSpec =
    s.curveData match
      case Some(cd) =>
        val newPoints = cd.points.grouped(3).flatMap {
          case Seq(px, py, pz) =>
            Seq((px - ox) * scale, (py - oy) * scale, (pz - oz) * scale)
          case _ => Seq.empty[Float]
        }.toVector
        s.copy(
          curveData = Some(cd.copy(points = Vector.from(newPoints))),
          x = (s.x - ox) * scale,
          y = (s.y - oy) * scale,
          z = (s.z - oz) * scale,
          size = s.size * scale
        )
      case None =>
        s.copy(
          x = (s.x - ox) * scale,
          y = (s.y - oy) * scale,
          z = (s.z - oz) * scale,
          size = s.size * scale
        )
