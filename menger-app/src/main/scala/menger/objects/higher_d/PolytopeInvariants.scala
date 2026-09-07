package menger.objects.higher_d

import menger.common.Const

/** One violated geometric invariant. `invariant` is a short, stable machine-readable tag
  * (e.g. `"euler-poincare"`); `message` is the human-readable detail. */
case class InvariantFinding(invariant: String, message: String)

/** Runtime, count-independent structural invariants every regular 4D polytope's `Mesh4D`
  * must satisfy, generalized from `Polytope4DContract`'s own test assertions (see
  * `menger-app/src/test/scala/menger/objects/higher_d/Polytope4DContract.scala`, lines
  * 43-52, 73-74, 96-103) so they can run against a `Mesh4D` whose expected V/E/F/C counts are
  * not known in advance -- e.g. a scene-generation agent's freshly compiled scene, where
  * nothing in the pipeline knows ahead of time that a tesseract should have exactly 16
  * vertices.
  *
  * Deliberately excludes the contract's own *type-specific* assertions (exact V/E/F/C
  * counts, `expectedCellShape`, exact edge length/vertex norm) -- those require knowing the
  * object's expected topology in advance, which this runtime check, by construction, cannot.
  */
object PolytopeInvariants:

  private val DefaultEpsilon = Const.epsilon * 100f

  /** Checks `mesh` against every count-independent invariant, returning one
    * [[InvariantFinding]] per violated invariant category (not one per offending vertex/face
    * -- a caller that needs per-element detail can re-derive it from the message). An empty
    * result means `mesh` is geometrically sound by every invariant this check knows about.
    */
  def check(mesh: Mesh4D, epsilon: Float = DefaultEpsilon): List[InvariantFinding] =
    List(
      checkFiniteCoordinates(mesh),
      checkNoDuplicateVertices(mesh),
      checkCommonSphere(mesh, epsilon),
      checkCenteredAtOrigin(mesh, epsilon),
      checkNoDegenerateFaces(mesh, epsilon),
      checkEulerPoincare(mesh)
    ).flatten

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

  private def checkNoDuplicateVertices(mesh: Mesh4D): Option[InvariantFinding] =
    val total    = mesh.vertices.size
    val distinct = mesh.vertices.distinct.size
    if distinct == total then None
    else Some(InvariantFinding(
      "no-duplicate-vertices",
      s"${total - distinct} duplicate vertex/vertices ($total total, $distinct distinct)"
    ))

  private def checkCommonSphere(mesh: Mesh4D, epsilon: Float): Option[InvariantFinding] =
    if mesh.vertices.isEmpty then None
    else
      val norms       = mesh.vertices.map(v => math.sqrt(v.len2.toDouble).toFloat)
      val (min, max)  = (norms.min, norms.max)
      if max - min <= epsilon then None
      else Some(InvariantFinding(
        "common-sphere",
        s"vertices are not all on a common sphere: norm range [$min, $max], " +
          s"spread ${max - min} exceeds epsilon $epsilon"
      ))

  private def checkCenteredAtOrigin(mesh: Mesh4D, epsilon: Float): Option[InvariantFinding] =
    if mesh.vertices.isEmpty then None
    else
      val centroid = mesh.vertices.reduce(_ + _) / mesh.vertices.size.toFloat
      val maxAbs   = centroid.toIndexedSeq.map(_.abs).max
      if maxAbs <= epsilon then None
      else Some(InvariantFinding(
        "centered-at-origin",
        s"centroid $centroid is not within epsilon $epsilon of the origin"
      ))

  // Review round 1 fix: was a hard `<= 0f` comparison, inconsistent with every sibling
  // check's use of `epsilon` -- a numerically-degenerate sliver face (area ~1e-9, not
  // exactly zero) silently passed.
  private def checkNoDegenerateFaces(mesh: Mesh4D, epsilon: Float): Option[InvariantFinding] =
    val degenerate = mesh.faces.zipWithIndex.filter { case (f, _) => f.area <= epsilon }
    if degenerate.isEmpty then None
    else Some(InvariantFinding(
      "no-degenerate-faces",
      s"${degenerate.size} of ${mesh.faces.size} faces have area at or below epsilon " +
        s"$epsilon (e.g. face ${degenerate.head._2})"
    ))

  private def checkEulerPoincare(mesh: Mesh4D): Option[InvariantFinding] =
    // `Mesh4D.cells` defaults to `Seq.empty` when a concrete mesh doesn't report cell
    // structure; the invariant only holds when cells (3D facets) are actually counted, so an
    // empty `cells` means "not applicable" rather than "violated".
    if mesh.cells.isEmpty then None
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
