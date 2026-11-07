package menger.spec

/** How a 4D object is viewed: the 4D shorthand [[Projection4DSpec]], or the general
  * [[ProjectionSpec]] for 4..8 dimensions (menger#162). Sealed so that engines match on both. */
sealed trait ProjectionLike

/**
 * 4D projection parameters for hypercube objects (tesseract, etc.).
 * Only applicable to object types where ObjectType.isHypercube returns true.
 */
case class Projection4DSpec(
  eyeW: Float = Projection4DSpec.DefaultEyeW,
  screenW: Float = Projection4DSpec.DefaultScreenW,
  rotXW: Degrees = Projection4DSpec.DefaultRotXW,
  rotYW: Degrees = Projection4DSpec.DefaultRotYW,
  rotZW: Degrees = Projection4DSpec.DefaultRotZW,
  // Scales the object's w coordinate before rotation and projection (menger#65): 0 flattens a
  // tesseract to a cube, animating 0 -> 1 grows it along w. Geometry, not view: see
  // ObjectSpec.withoutView.
  wScale: Float = Projection4DSpec.DefaultWScale
) extends ProjectionLike:
  require(eyeW > screenW, s"eyeW ($eyeW) must be greater than screenW ($screenW)")
  require(eyeW > 0 && screenW > 0, "eyeW and screenW must be positive")
  require(wScale >= 0f && !wScale.isInfinite, s"wScale must be finite and >= 0, got $wScale")

object Projection4DSpec:
  val DefaultEyeW: Float = 3.0f
  val DefaultScreenW: Float = 1.5f
  val DefaultRotXW: Degrees = Degrees(15f)
  val DefaultRotYW: Degrees = Degrees(10f)
  val DefaultRotZW: Degrees = Degrees.Zero
  val DefaultWScale: Float = 1f

  val default: Projection4DSpec = Projection4DSpec()

/** A rotation of one coordinate plane, e.g. `PlaneRotation("xv", 30f)`: for a positive angle
  * the second axis of the plane turns towards the first (v towards x), as w turns towards x
  * for a positive `rotXW`. */
case class PlaneRotation(plane: PlaneName, angle: Degrees):
  require(angle.value.isFinite, s"angle of plane '$plane' must be finite, got ${angle.value}")

/** The eye and the screen along the axis one projection stage drops. */
case class ProjectionStage(eye: Float, screen: Float):
  require(eye.isFinite, s"eye must be finite, got $eye")
  require(screen.isFinite, s"screen must be finite, got $screen")
  require(eye > screen, s"eye ($eye) must be greater than screen ($screen)")
  require(eye > 0 && screen > 0, "eye and screen must be positive")

/** The view of a 4D object lifted into `dimension` (4..8) dimensions: the object lies in the
  * x y z w subspace, `planes` turn it in the order written (the first turns first), and a
  * perspective chain projects it dimension -> dimension - 1 -> ... -> 3 (menger#162).
  * `stages` has one eye and screen per projection stage, the first dropping the highest axis;
  * left empty, every stage uses `eye` and `screen`. */
case class ProjectionSpec(
  dimension: Int,
  planes: List[PlaneRotation] = Nil,
  eye: Float = ProjectionSpec.DefaultEye,
  screen: Float = ProjectionSpec.DefaultScreen,
  stages: List[ProjectionStage] = Nil
) extends ProjectionLike:
  require(eye.isFinite, s"eye must be finite, got $eye")
  require(screen.isFinite, s"screen must be finite, got $screen")
  require(
    dimension >= ProjectionSpec.MinDimension && dimension <= ProjectionSpec.MaxDimension,
    s"dimension must be between ${ProjectionSpec.MinDimension} and " +
      s"${ProjectionSpec.MaxDimension}, is $dimension"
  )
  require(eye > screen, s"eye ($eye) must be greater than screen ($screen)")
  require(eye > 0 && screen > 0, "eye and screen must be positive")
  planes.map(_.plane).foreach { plane =>
    val missing = PlaneName.axisLetters(plane.highestAxis)
    require(
      plane.highestAxis < dimension,
      s"plane '$plane' needs axis '$missing', but dimension $dimension has only " +
        PlaneName.axisLetters.take(dimension).mkString(" ")
    )
  }
  require(
    stages.isEmpty || stages.size == stageCount,
    s"dimension $dimension needs $stageCount projection stages, got ${stages.size}"
  )

  def stageCount: Int = dimension - ProjectionSpec.TargetDimension

  /** The (eye, screen) of every stage, the first dropping the highest axis. */
  def stagePairs: List[ProjectionStage] =
    if stages.isEmpty then List.fill(stageCount)(ProjectionStage(eye, screen)) else stages

object ProjectionSpec:
  val MinDimension: Int = 4
  val MaxDimension: Int = 8
  val TargetDimension: Int = 3
  val DefaultEye: Float = Projection4DSpec.DefaultEyeW
  val DefaultScreen: Float = Projection4DSpec.DefaultScreenW

  /** A fractional-level sponge is a pair of GPU-projected meshes, which the CPU chain does not
    * build. */
  def wholeLevelMessage(objectType: String, level: Float): String =
    s"$objectType with a ProjectionSpec needs a whole level, got $level"

/** A coordinate plane named by its two axis letters, e.g. "xw" or "xv". The axes are
  * x y z w v u t s (5D v, 6D u, 7D t, 8D s). */
case class PlaneName(letters: String):
  require(letters.length == 2, s"plane '$letters' must name exactly two axes, like \"xw\"")
  require(
    letters.forall(PlaneName.axisLetters.contains),
    s"plane '$letters' uses an unknown axis letter; the axes are " +
      PlaneName.axisLetters.mkString(" ")
  )
  require(letters(0) != letters(1), s"plane '$letters' names the same axis twice")

  def firstAxis: Int = PlaneName.axisLetters.indexOf(letters(0))
  def secondAxis: Int = PlaneName.axisLetters.indexOf(letters(1))
  def highestAxis: Int = math.max(firstAxis, secondAxis)

  override def toString: String = letters

object PlaneName:
  val axisLetters: IndexedSeq[Char] = "xyzwvuts"

  given Conversion[String, PlaneName] = PlaneName(_)
