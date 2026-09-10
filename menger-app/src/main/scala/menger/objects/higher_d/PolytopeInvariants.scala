package menger.objects.higher_d

import menger.common.Const

/** One violated geometric invariant. `invariant` is a short, stable machine-readable tag
  * (e.g. `"euler-poincare"`); `message` is the human-readable detail. */
case class InvariantFinding(invariant: String, message: String)

/** Runtime, count-independent structural invariants a 4D `Mesh4D` must satisfy, generalized
  * from `Polytope4DContract`'s own test assertions (see
  * `menger-app/src/test/scala/menger/objects/higher_d/Polytope4DContract.scala`, lines
  * 43-52, 73-74, 96-103) so they can run against a `Mesh4D` whose expected V/E/F/C counts are
  * not known in advance -- e.g. a scene-generation agent's freshly compiled scene, where
  * nothing in the pipeline knows ahead of time that a tesseract should have exactly 16
  * vertices.
  *
  * Deliberately excludes the contract's own *type-specific* assertions (exact V/E/F/C
  * counts, `expectedCellShape`, exact edge length/vertex norm) -- those require knowing the
  * object's expected topology in advance, which this runtime check, by construction, cannot.
  *
  * ==Two invariant classes==
  *
  * Only [[shapeIndependentFindings]] -- finite coordinates and vertex distinctness -- holds
  * for *every* `Mesh4D`. The rest (`common-sphere`, `centered-at-origin`,
  * `no-degenerate-faces`, `euler-poincare`) are properties of a *regular* polytope and are
  * false for a 4D fractal by construction: a `TesseractSponge` at level 1 has vertex norms
  * spanning [0.33, 1.0], so `common-sphere` reports every valid sponge scene as defective.
  * Review round 2 fix: `check` now dispatches on [[Fractal4D]] and runs only the
  * shape-independent subset for fractals, rather than reporting the renderer's own shipped
  * corpus exemplars (`FractalWithHDR.scala`, `EnvMapVideoSponge.scala`) as lint findings.
  *
  * Every epsilon comparison is *relative to the mesh's own scale* (review round 2): the
  * absolute `DefaultEpsilon` flagged a legitimately small object -- `Tesseract(size = 0.02f)`
  * has faces of area 4e-4, below a 1e-3 absolute bound -- as entirely degenerate.
  */
object PolytopeInvariants:

  private val DefaultEpsilon = Const.epsilon * 100f

  /** Checks `mesh` against every invariant that applies to it, returning one
    * [[InvariantFinding]] per violated invariant category (not one per offending vertex/face
    * -- a caller that needs per-element detail can re-derive it from the message). An empty
    * result means `mesh` is geometrically sound by every invariant this check knows about.
    *
    * A non-finite coordinate short-circuits the rest: NaN/Inf makes every downstream norm,
    * centroid and area comparison meaningless, so continuing would emit three more findings
    * whose messages are all `NaN` (review round 2).
    */
  def check(mesh: Mesh4D, epsilon: Float = DefaultEpsilon): List[InvariantFinding] =
    checkFiniteCoordinates(mesh) match
      case Some(finding) => List(finding)
      case None          =>
        val shapeIndependent = shapeIndependentFindings(mesh, epsilon)
        mesh match
          case _: Fractal4D => shapeIndependent
          case _            => shapeIndependent ::: regularPolytopeFindings(mesh, epsilon)

  /** Invariants true of any `Mesh4D`, fractal or regular. */
  private def shapeIndependentFindings(mesh: Mesh4D, epsilon: Float): List[InvariantFinding] =
    List(checkNoDuplicateVertices(mesh, epsilon)).flatten

  /** Invariants true only of a *regular* 4D polytope -- all vertices on a common sphere,
    * centroid at the origin, no degenerate face, Euler-Poincare characteristic zero. */
  private def regularPolytopeFindings(mesh: Mesh4D, epsilon: Float): List[InvariantFinding] =
    List(
      checkCommonSphere(mesh, epsilon),
      checkCenteredAtOrigin(mesh, epsilon),
      checkNoDegenerateFaces(mesh, epsilon),
      checkEulerPoincare(mesh)
    ).flatten

  /** The mesh's own linear scale, used to make every epsilon comparison relative rather than
    * absolute. `1f` for an empty or degenerate-at-origin mesh so the comparisons fall back to
    * the absolute bound instead of collapsing to zero tolerance. */
  private def scaleOf(mesh: Mesh4D): Float =
    val maxNorm = mesh.vertices.map(v => math.sqrt(v.len2.toDouble).toFloat).maxOption.getOrElse(0f)
    if maxNorm > 0f then maxNorm else 1f

  private def checkFiniteCoordinates(mesh: Mesh4D): Option[InvariantFinding] =
    val offending = mesh.vertices.zipWithIndex.filter { case (v, _) =>
      v.toIndexedSeq.exists(c => c.isNaN || c.isInfinite)
    }
    if offending.isEmpty then None
    else Some(InvariantFinding(
      "finite-vertex-coordinates",
      s"${offending.size} of ${mesh.vertices.size} vertices have a NaN/Inf coordinate " +
        s"(e.g. vertex ${offending.head._2}: ${offending.head._1})"
    ))

  // Review round 2 fix: was `mesh.vertices.distinct`, an exact float equality while every
  // sibling check used `epsilon` -- two vertices 1e-9 apart counted as distinct and passed.
  // Quantizing to an epsilon-sized grid (scaled by the mesh's own size, as everywhere else)
  // makes "duplicate" mean the same thing here as "degenerate" does for a face.
  private def checkNoDuplicateVertices(mesh: Mesh4D, epsilon: Float): Option[InvariantFinding] =
    val tolerance = epsilon * scaleOf(mesh)
    val total     = mesh.vertices.size
    val quantized = mesh.vertices.map(v => v.toIndexedSeq.map(c => math.round(c / tolerance)))
    val distinct  = quantized.distinct.size
    if distinct == total then None
    else Some(InvariantFinding(
      "no-duplicate-vertices",
      s"${total - distinct} duplicate vertex/vertices within $tolerance " +
        s"($total total, $distinct distinct)"
    ))

  private def checkCommonSphere(mesh: Mesh4D, epsilon: Float): Option[InvariantFinding] =
    if mesh.vertices.isEmpty then None
    else
      val norms      = mesh.vertices.map(v => math.sqrt(v.len2.toDouble).toFloat)
      val (min, max) = (norms.min, norms.max)
      val tolerance  = epsilon * scaleOf(mesh)
      if max - min <= tolerance then None
      else Some(InvariantFinding(
        "common-sphere",
        s"vertices are not all on a common sphere: norm range [$min, $max], " +
          s"spread ${max - min} exceeds tolerance $tolerance"
      ))

  private def checkCenteredAtOrigin(mesh: Mesh4D, epsilon: Float): Option[InvariantFinding] =
    if mesh.vertices.isEmpty then None
    else
      val centroid  = mesh.vertices.reduce(_ + _) / mesh.vertices.size.toFloat
      val maxAbs    = centroid.toIndexedSeq.map(_.abs).max
      val tolerance = epsilon * scaleOf(mesh)
      if maxAbs <= tolerance then None
      else Some(InvariantFinding(
        "centered-at-origin",
        s"centroid $centroid is not within tolerance $tolerance of the origin"
      ))

  // Review round 1 fix: was a hard `<= 0f` comparison, inconsistent with every sibling
  // check's use of `epsilon` -- a numerically-degenerate sliver face (area ~1e-9, not
  // exactly zero) silently passed.
  // Review round 2 fix: the epsilon was absolute, so every face of a legitimately small
  // object (`Tesseract(size = 0.02f)`: face area 4e-4 < 1e-3) was reported degenerate.
  // Compared against the mesh's own largest face instead.
  private def checkNoDegenerateFaces(mesh: Mesh4D, epsilon: Float): Option[InvariantFinding] =
    val areaRef    = mesh.faces.map(_.area).maxOption.getOrElse(0f)
    val tolerance  = if areaRef > 0f then epsilon * areaRef else epsilon
    val degenerate = mesh.faces.zipWithIndex.filter { case (f, _) => f.area <= tolerance }
    if degenerate.isEmpty then None
    else Some(InvariantFinding(
      "no-degenerate-faces",
      s"${degenerate.size} of ${mesh.faces.size} faces have area at or below tolerance " +
        s"$tolerance (e.g. face ${degenerate.head._2})"
    ))

  private def checkEulerPoincare(mesh: Mesh4D): Option[InvariantFinding] =
    // `Mesh4D.cells` defaults to `Seq.empty` when a concrete mesh doesn't report cell
    // structure; the invariant only holds when cells (3D facets) are actually counted, so an
    // empty `cells` means "not applicable" rather than "violated". The same applies when a
    // mesh reports cells but no faces/edges -- the characteristic is not computable from a
    // partial complex (review round 2).
    if mesh.cells.isEmpty || mesh.faces.isEmpty || mesh.edges.isEmpty then None
    else
      val v   = mesh.vertices.size
      val e   = mesh.edges.size
      val f   = mesh.faces.size
      val c   = mesh.cells.size
      val chi = v - e + f - c
      if chi == 0 then None
      else Some(InvariantFinding(
        "euler-poincare",
        s"V-E+F-C = $v-$e+$f-$c = $chi, expected 0"
      ))
