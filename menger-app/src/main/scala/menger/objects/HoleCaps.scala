package menger.objects

import menger.common.TriangleMeshData

/** The hole caps of a fractional sponge level (usability review 2026-09, session 2, F35).
  * Going from level n to n+1, every exposed square of the level-n surface loses exactly its
  * centre ninth. A cap is that centre ninth: the quad shrunk to a third about its own centre.
  * Fading the caps out fades the new holes in, and no cap overlaps a level-(n+1) face, unlike
  * the whole level-n "skin" this replaces. */
object HoleCaps:

  private val CapFraction = 1f / 3f
  private val VerticesPerQuad = 4
  private val PositionOffset = 0
  private val UvOffset = 6
  private val Dimensions = 3
  private val UvDimensions = 2

  /** `mesh` must consist of quads of 4 consecutive vertices, as `Face.toTriangleMesh` and
    * everything merged from it emit them. */
  def of(mesh: TriangleMeshData): TriangleMeshData =
    require(
      mesh.vertexStride == TriangleMeshData.DefaultVertexStride,
      s"hole caps need pos+normal+uv vertices, got stride ${mesh.vertexStride}"
    )
    require(
      mesh.numVertices % VerticesPerQuad == 0,
      s"hole caps need a mesh of quads, got ${mesh.numVertices} vertices"
    )
    val stride = mesh.vertexStride
    val source = mesh.vertices
    val caps = source.clone()
    for
      quad <- 0 until mesh.numVertices / VerticesPerQuad
      axis <- 0 until Dimensions
    do
      val offsets = (0 until VerticesPerQuad).map(i =>
        (quad * VerticesPerQuad + i) * stride + PositionOffset + axis
      )
      val centre = offsets.map(source(_)).sum / VerticesPerQuad
      offsets.foreach(k => caps(k) = centre + (source(k) - centre) * CapFraction)
    for
      vertex <- 0 until mesh.numVertices
      uv <- 0 until UvDimensions
    do
      val k = vertex * stride + UvOffset + uv
      caps(k) = CapFraction + source(k) * CapFraction
    TriangleMeshData(caps, mesh.indices, stride)
