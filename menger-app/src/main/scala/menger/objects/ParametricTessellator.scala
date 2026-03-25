package menger.objects

import com.typesafe.scalalogging.LazyLogging
import menger.common.TriangleMeshData

object ParametricTessellator extends LazyLogging:

  private val MemoryWarningThreshold = 1_000_000

  def tessellate(
    f: (Float, Float) => (Float, Float, Float),
    uRange: (Float, Float),
    vRange: (Float, Float),
    uSteps: Int,
    vSteps: Int,
    closedU: Boolean,
    closedV: Boolean
  ): TriangleMeshData =
    require(uSteps >= 1, s"uSteps must be >= 1, got $uSteps")
    require(vSteps >= 1, s"vSteps must be >= 1, got $vSteps")

    val totalCells = uSteps.toLong * vSteps.toLong
    if totalCells > MemoryWarningThreshold then
      val approxMB = totalCells * 8 * 4 / (1024 * 1024)
      logger.warn(
        "Parametric surface tessellation is very high resolution " +
        s"($uSteps x $vSteps = $totalCells grid cells). " +
        s"This will use approximately $approxMB MB of GPU memory. " +
        "Consider reducing resolution."
      )

    val (uMin, uMax) = uRange
    val (vMin, vMax) = vRange
    val du = (uMax - uMin) / uSteps
    val dv = (vMax - vMin) / vSteps

    // Number of unique vertices depends on closure
    val uVerts = if closedU then uSteps else uSteps + 1
    val vVerts = if closedV then vSteps else vSteps + 1
    val numVerts = uVerts * vVerts

    val vertices = new Array[Float](numVerts * 8)
    val epsilon = 1e-4f * math.max(math.abs(uMax - uMin), math.abs(vMax - vMin))

    for j <- 0 until vVerts; i <- 0 until uVerts do
      val u = uMin + i * du
      val v = vMin + j * dv
      val (px, py, pz) = f(u, v)

      // Finite difference normals
      val (dxu, dyu, dzu) =
        val (x1, y1, z1) = f(u + epsilon, v)
        val (x0, y0, z0) = f(u - epsilon, v)
        ((x1 - x0) / (2 * epsilon), (y1 - y0) / (2 * epsilon), (z1 - z0) / (2 * epsilon))
      val (dxv, dyv, dzv) =
        val (x1, y1, z1) = f(u, v + epsilon)
        val (x0, y0, z0) = f(u, v - epsilon)
        ((x1 - x0) / (2 * epsilon), (y1 - y0) / (2 * epsilon), (z1 - z0) / (2 * epsilon))

      // cross(du, dv)
      val crossX = dyu * dzv - dzu * dyv
      val crossY = dzu * dxv - dxu * dzv
      val crossZ = dxu * dyv - dyu * dxv
      val len = math.sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ).toFloat

      val (nx, ny, nz) =
        if len < 1e-8f then
          // Degenerate: use position as normal, or fallback to (0,1,0)
          val pLen = math.sqrt(px * px + py * py + pz * pz).toFloat
          if pLen > 1e-8f then (px / pLen, py / pLen, pz / pLen)
          else (0f, 1f, 0f)
        else
          (crossX / len, crossY / len, crossZ / len)

      // UV coordinates normalized to [0, 1]
      val texU = if uMax == uMin then 0f else (u - uMin) / (uMax - uMin)
      val texV = if vMax == vMin then 0f else (v - vMin) / (vMax - vMin)

      val idx = (j * uVerts + i) * 8
      vertices(idx) = px; vertices(idx + 1) = py; vertices(idx + 2) = pz
      vertices(idx + 3) = nx; vertices(idx + 4) = ny; vertices(idx + 5) = nz
      vertices(idx + 6) = texU; vertices(idx + 7) = texV

    // Index generation with seam welding
    val indices = Array.newBuilder[Int]
    for j <- 0 until vSteps; i <- 0 until uSteps do
      val i0 = i
      val i1 = if closedU && i + 1 == uSteps then 0 else i + 1
      val j0 = j
      val j1 = if closedV && j + 1 == vSteps then 0 else j + 1

      val topLeft     = j0 * uVerts + i0
      val topRight    = j0 * uVerts + i1
      val bottomLeft  = j1 * uVerts + i0
      val bottomRight = j1 * uVerts + i1

      // Triangle 1
      indices += topLeft; indices += bottomLeft; indices += bottomRight
      // Triangle 2
      indices += topLeft; indices += bottomRight; indices += topRight

    TriangleMeshData(vertices, indices.result(), vertexStride = 8)
