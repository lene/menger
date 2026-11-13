package menger.objects.higher_d

import java.util.Arrays
import java.util.concurrent.atomic.AtomicInteger

import scala.annotation.tailrec
import scala.collection.concurrent.TrieMap

/** The two sponge rules, and the object type each is drawn as (the one place that maps them). */
enum NdSpongeVariant(val objectTypeName: String):
  case Volume extends NdSpongeVariant("n-sponge-volume")
  case Surface extends NdSpongeVariant("n-sponge-surface")

object NdSpongeVariant:
  def fromObjectType(objectType: String): Option[NdSpongeVariant] =
    values.find(_.objectTypeName == objectType)

/** Menger sponges of 4 to 8 dimensions as sets of square faces (menger#166, decision card E2).
  *
  * The n-cube [-0.5, 0.5]^n is split 3 times per axis, level times over. `maxCentreCoordinates`
  * (k) is the most coordinates a kept sub-cube may have in the middle third of its parent: 1
  * gives the 4D `TesseractSponge` and `TesseractSponge2`, n - 2 removes only the middle cell and
  * its axis neighbours, and n - 1 (the largest value) removes only the middle sub-cube of each
  * step.
  *
  * Faces live on an integer grid in which a finest cell has half-size 1 and the cube spans
  * +-3^level, and are held as packed Long keys: two axis indices, then the n centre coordinates.
  * Sorting the keys finds the faces cells share and drops duplicates. */
object NdSponge:
  val MinDimension: Int = 4
  val MaxDimension: Int = 8
  val DefaultMaxCentreCoordinates: Int = 1

  private val Base = 3
  private val AxisIndexBits = 3
  // the sign bit stays clear, so keys sort the same as unsigned and -1 can mark "no face"
  private val KeyBitBudget = 63
  private val NoFace = -1L
  // corners of a square around its centre, once around: (-,-), (+,-), (+,+), (-,+)
  private val CornerSignsI = Array(-1, 1, 1, -1)
  private val CornerSignsJ = Array(-1, -1, 1, 1)
  // the nine sub-squares of a face split 3 x 3, row by row
  private val FlatParts = Base * Base
  // per normal axis: the face's two edge axes, both sides of the hole; one direction, toward
  // the centre
  private val WallsPerNormal = 4
  private val OuterShift = 2
  private val InPlaneAxes = 2
  // the volume recurrence also counts the 3D sponge, which the builds do not accept
  private val MinSpongeDimension = 3
  // the first axis of a face is stored in AxisIndexBits bits, the last axis index is n - 1
  require(
    MaxDimension <= (1 << AxisIndexBits),
    s"a face key stores axis indices in $AxisIndexBits bits, too few for $MaxDimension dimensions"
  )
  // 2 * 3^39 is the largest coordinate range that a Long holds
  private val MaxExactLevel = 39

  def apply(
    variant: NdSpongeVariant, n: Int, level: Int, k: Int = DefaultMaxCentreCoordinates
  ): IndexedMeshN =
    variant match
      case NdSpongeVariant.Volume => volume(n, level, k)
      case NdSpongeVariant.Surface => surface(n, level, k)

  /** The offsets (-1, 0 or 1 per axis, the sub-cube's third of its parent) of the sub-cubes kept
    * in one step. */
  def keptOffsets(n: Int, k: Int): IndexedSeq[IndexedSeq[Int]] =
    requireDimensionAndRange(n, k)
    (0 until n).foldLeft(IndexedSeq(IndexedSeq.empty[Int])) { (prefixes, _) =>
      for prefix <- prefixes; offset <- -1 to 1 yield prefix :+ offset
    }.filter(_.count(_ == 0) <= k)

  /** The centres of the kept finest cells, n integers per cell, one cell after the other. */
  def keptCells(n: Int, level: Int, k: Int): Array[Int] =
    requireKeyFits(n, level, k)
    requireBuildable(keptCellCandidates(n, level, k), cellLimit(n), n, level, k, "kept cells")
    val offsets = keptOffsets(n, k).flatten.toArray
    val offsetCount = offsets.length / n
    (0 until level).foldLeft(Array.fill(n)(0)) { (cells, step) =>
      val scale = 2 * power(Base, level - step - 1)
      Array.tabulate(cells.length / n * offsetCount * n) { index =>
        val axis = index % n
        val offset = index / n % offsetCount
        val cell = index / n / offsetCount
        cells(cell * n + axis) + offsets(offset * n + axis) * scale
      }
    }

  /** Faces between a kept and a removed sub-cube: a face all 2^(n-2) cells around it keep is
    * inside the sponge. */
  def volume(n: Int, level: Int, k: Int = DefaultMaxCentreCoordinates): IndexedMeshN =
    val keys = keysFor(n, level, k)
    requireBuildableFaces(NdSpongeVariant.Volume, n, level, k)
    val cells = keptCells(n, level, k)
    val cellCount = cells.length / n
    val templates = faceTemplates(keys)
    val raw = new Array[Long](cellCount * templates.length)
    (0 until cellCount).foreach { cell =>
      val base = keys.cellBase(cells, cell)
      templates.indices.foreach(t => raw(cell * templates.length + t) = base + templates(t))
    }
    Arrays.sort(raw)
    val interior = 1 << (n - 2)
    meshOf(keys, uniqueRuns(raw, _ < interior))

  /** The surface rule S: every face splits into nine squares, the middle one is replaced by walls
    * around its hole along each of the face's normals, and faces outside the sponge are dropped. */
  def surface(n: Int, level: Int, k: Int = DefaultMaxCentreCoordinates): IndexedMeshN =
    val keys = keysFor(n, level, k)
    requireBuildableFaces(NdSpongeVariant.Surface, n, level, k)
    meshOf(keys, surfaceKeys(keys, level, k))

  private def surfaceKeys(keys: FaceKeys, level: Int, k: Int): Array[Long] =
    (1 to level).foldLeft(cubeFaces(keys))((faces, step) => refine(keys, level, faces, step, k))

  // Work done on the calling thread, for tests that show a path builds no mesh and counts no
  // more than it must (menger#183): they cannot read a count that other threads move.
  private val meshesBuilt = ThreadLocal.withInitial(() => AtomicInteger(0))
  private val surfaceCountRequests = ThreadLocal.withInitial(() => AtomicInteger(0))

  /** The meshes `volume` and `surface` built on this thread so far. */
  private[menger] def meshesBuiltOnThisThread: Int = meshesBuilt.get.get

  /** The calls of `faceCount` for a surface on this thread so far, cached ones included. */
  private[menger] def surfaceCountRequestsOnThisThread: Int = surfaceCountRequests.get.get

  /** A surface count computed at most once, however many threads ask for it at the same time:
    * a count takes seconds and some hundred MB. */
  private final class CountOnce(compute: () => Long):
    lazy val value: Long = compute()

  // (n, level, k) -> faces of the surface sponge; a few hundred Longs at most
  private val surfaceCounts = TrieMap.empty[(Int, Int, Int), CountOnce]

  /** The surface's faces counted on the face keys alone: no vertices, no mesh. */
  private def surfaceFaceCount(n: Int, level: Int, k: Int): Long =
    surfaceCountRequests.get.incrementAndGet()
    val once = CountOnce { () =>
      val keys = keysFor(n, level, k)
      requireBuildableFaces(NdSpongeVariant.Surface, n, level, k)
      surfaceKeys(keys, level, k).length.toLong
    }
    surfaceCounts.putIfAbsent((n, level, k), once).getOrElse(once).value

  /** A cheap bound on the number of faces, which is also the number of face keys the build
    * handles before merging: for the volume the faces of all kept cells, for the surface the
    * slots of the last step. Saturates at Long.MaxValue. */
  def upperBoundFaces(variant: NdSpongeVariant, n: Int, level: Int, k: Int): Long =
    requireDimensionAndRange(n, k)
    require(level >= 0, s"level must not be negative, got $level")
    val perStep = variant match
      case NdSpongeVariant.Volume => keptPerStep(n, k)
      case NdSpongeVariant.Surface => surfaceChildrenPerFace(n)
    saturatingMultiply(NCube.faceCount(n).toLong, saturatingPower(perStep, level))

  /** The number of faces of the sponge, without building a mesh. The volume is counted exactly
    * by `volumeFaceCount`, the surface by refining face keys only (memoised).
    *
    * The two differ at the limits. The surface path throws an IllegalArgumentException for a
    * level past the key width or the array limit (what `isBuildable` says no to), while the
    * volume path returns its count. When the upper bound saturates (a huge level) the result is
    * Long.MaxValue at once: an upper limit, not an exact count. Callers guard with
    * `isBuildable`. */
  def faceCount(variant: NdSpongeVariant, n: Int, level: Int, k: Int): Long =
    requireDimensionAndRange(n, k)
    require(level >= 0, s"level must not be negative, got $level")
    if upperBoundFaces(variant, n, level, k) == Long.MaxValue then Long.MaxValue
    else variant match
      case NdSpongeVariant.Volume => volumeFaceCount(n, level, k).min(BigInt(Long.MaxValue)).toLong
      case NdSpongeVariant.Surface => surfaceFaceCount(n, level, k)

  /** The faces of the volume sponge by a recurrence over the levels, with no arrays of faces.
    *
    * A face is a unit square of the finest grid: it spans two "in-plane" axes and has n - 2
    * "normal" axes, on each of which it sits between two neighbouring cells or on the outer
    * boundary. The mesh keeps it when some cell around it is kept and not all 2^(n-2) are.
    *
    * On an interior normal axis, the base-3 digits of the two neighbouring cells agree down to
    * the "split level", where they are (0, 1) or (1, 2), and below it they are (2, 0): at the
    * split level exactly one side has the middle digit, above it both do or neither, below it
    * neither. With, at one level, c in-plane axes with the middle digit, a unsplit normal axes
    * with it and m normal axes splitting there, some cell is kept iff a + c <= k at every
    * level, and all are kept iff a + c + m <= k at every level. `unsplitWays` counts the digit
    * choices level by level; boundary normal axes never have a middle digit. */
  private[higher_d] def volumeFaceCount(n: Int, level: Int, k: Int): BigInt =
    require(
      n >= MinSpongeDimension && k >= 1 && k <= n - 1 && level >= 0,
      s"volumeFaceCount needs n >= $MinSpongeDimension, 1 <= k <= n - 1 and level >= 0, " +
        s"got n=$n, level=$level, k=$k"
    )
    val normals = n - InPlaneAxes
    val anyKept = unsplitWays(level, normals, k, strict = false)
    val allKept = unsplitWays(level, normals, k, strict = true)
    val perAxisPair = (0 to normals).map { boundary =>
      val allKeptFaces = if boundary == 0 then allKept(normals) else BigInt(0)
      binomialBig(normals, boundary) * twoTo(boundary) *
        (anyKept(normals - boundary) - allKeptFaces)
    }.sum
    binomialBig(n, InPlaneAxes) * perAxisPair

  /** For each number u of unsplit interior normal axes, the ways to choose the digits of levels
    * 1 to `level` of a face. `strict` counts the choices that keep every cell around the face. */
  private def unsplitWays(
    level: Int, normals: Int, k: Int, strict: Boolean
  ): IndexedSeq[BigInt] =
    val beyondFinest =
      IndexedSeq.tabulate(normals + 1)(u => if u == 0 then BigInt(1) else BigInt(0))
    (level to 1 by -1).foldLeft(beyondFinest) { (next, lv) =>
      IndexedSeq.tabulate(normals + 1)(waysAtLevel(_, lv == level, k, strict, next))
    }

  private def waysAtLevel(
    unsplit: Int, finest: Boolean, k: Int, strict: Boolean, next: IndexedSeq[BigInt]
  ): BigInt =
    val terms = for
      splitting <- 0 to unsplit
      if !finest || splitting == unsplit
      middle <- 0 to unsplit - splitting
      inPlane <- 0 to InPlaneAxes
      if middle + inPlane + (if strict then splitting else 0) <= k
    yield
      val rest = unsplit - splitting
      binomialBig(unsplit, splitting) * twoTo(splitting) *
        binomialBig(rest, middle) * twoTo(rest - middle) *
        binomialBig(InPlaneAxes, inPlane) * twoTo(InPlaneAxes - inPlane) *
        next(rest)
    terms.sum

  private def binomialBig(n: Int, k: Int): BigInt = BigInt(binomial(n, k))

  private def twoTo(exponent: Int): BigInt = BigInt(1) << exponent

  /** What `refine` reserves for one face: the 8 outer squares of the 3 x 3 split, and for each
    * of the n - 2 normals up to 4 walls (toward the centre, 2 edge axes, 2 sides) for the middle
    * square. The surface starts from C(n,2) 2^(n-2) faces and multiplies by this once per level,
    * so no step has more faces than this bound allows. */
  private def surfaceChildrenPerFace(n: Int): Long =
    FlatParts - 1L + WallsPerNormal.toLong * (n - 2)

  /** Whether `apply` can build this sponge at all: the arguments are in range, the face key fits
    * a Long and every array the build allocates fits an Int. Says nothing about memory. The
    * `require`s of the build check the same predicates. */
  def isBuildable(variant: NdSpongeVariant, n: Int, level: Int, k: Int): Boolean =
    inRange(n, k) && level >= 0 && keyFits(n, level) &&
      upperBoundFaces(variant, n, level, k) <= FaceLimit &&
      (variant != NdSpongeVariant.Volume || keptCellCandidates(n, level, k) <= cellLimit(n))

  private val FaceLimit: Int = Int.MaxValue / IndexedMeshN.CornersPerFace

  private def cellLimit(n: Int): Int = Int.MaxValue / n

  private def keptCellCandidates(n: Int, level: Int, k: Int): Long =
    saturatingPower(keptPerStep(n, k), level)

  private def requireBuildableFaces(variant: NdSpongeVariant, n: Int, level: Int, k: Int): Unit =
    requireBuildable(
      upperBoundFaces(variant, n, level, k), FaceLimit, n, level, k, s"$variant faces"
    )

  private def requireBuildable(
    candidates: Long, limit: Int, n: Int, level: Int, k: Int, what: String
  ): Unit =
    require(
      candidates <= limit,
      s"a ${n}D sponge of level $level (k=$k) is too large to build: up to $candidates $what, " +
        s"more than an array can hold ($limit)"
    )

  private def inRange(n: Int, k: Int): Boolean =
    n >= MinDimension && n <= MaxDimension && k >= 1 && k <= n - 1

  private def requireDimensionAndRange(n: Int, k: Int): Unit =
    require(
      n >= MinDimension && n <= MaxDimension,
      s"dimension must be between $MinDimension and $MaxDimension, is $n"
    )
    require(
      k >= 1 && k <= n - 1,
      s"maxCentreCoordinates must be in the range 1..${n - 1} for dimension $n, is $k"
    )

  // coordinates 0..2*3^level need this many bits
  private def coordinateBits(level: Int): Int =
    if level > MaxExactLevel then java.lang.Long.SIZE
    else (BigInt(Base).pow(level) * 2).bitLength

  // ponytail: n * bits + 6 <= 63 caps the level at 3 for n = 8, 4 for n = 7, 5 for n = 6,
  // 6 for n = 5 and 8 for n = 4; wider keys would need two Longs per face.
  private def keyFits(n: Int, level: Int): Boolean =
    n * coordinateBits(level) + 2 * AxisIndexBits <= KeyBitBudget

  private def requireKeyFits(n: Int, level: Int, k: Int): Unit =
    requireDimensionAndRange(n, k)
    require(level >= 0, s"level must not be negative, got $level")
    require(
      keyFits(n, level),
      s"N=$n, L=$level: the face key is too wide ($n coordinates of ${coordinateBits(level)} " +
        s"bits and two axis indices do not fit in $KeyBitBudget bits)"
    )

  private[higher_d] def keysFor(n: Int, level: Int, k: Int): FaceKeys =
    requireKeyFits(n, level, k)
    FaceKeys(n, coordinateBits(level), power(Base, level))

  private def power(base: Int, exponent: Int): Int =
    (0 until exponent).foldLeft(1)((result, _) => result * base)

  private def binomial(n: Int, k: Int): Long =
    (1 to k).foldLeft(1L)((acc, i) => acc * (n - k + i) / i)

  /** Sub-cubes with j middle coordinates: C(n,j) places, 2^(n-j) outer thirds on the others. */
  private def keptPerStep(n: Int, k: Int): Long =
    (0 to k).map(j => binomial(n, j) * (1L << (n - j))).sum

  private def saturatingMultiply(a: Long, b: Long): Long =
    if a == 0 || b == 0 then 0L
    else if a > Long.MaxValue / b then Long.MaxValue
    else a * b

  // stops multiplying once the product has saturated: base is at least 2, level may be huge
  @tailrec
  private def saturatingPower(base: Long, exponent: Int, product: Long = 1L): Long =
    if exponent == 0 || product == Long.MaxValue then product
    else saturatingPower(base, exponent - 1, saturatingMultiply(product, base))

  /** A face key, its parts added to or taken from the key of the cell it belongs to: the key of
    * a face is the key of its centre point plus the axis pair. */
  private[higher_d] final case class FaceKeys(n: Int, bits: Int, extent: Int):
    private val coordinateMask = (1L << bits) - 1
    private val coordinatesBits = n * bits
    private val coordinatesMask = (1L << coordinatesBits) - 1

    def shift(axis: Int): Int = (n - 1 - axis) * bits

    def unit(axis: Int): Long = 1L << shift(axis)

    def header(i: Int, j: Int): Long = ((i.toLong << AxisIndexBits) | j) << coordinatesBits

    def encode(i: Int, j: Int, centre: Array[Int]): Long =
      (0 until n).foldLeft(header(i, j))((key, axis) =>
        key | ((centre(axis) + extent).toLong << shift(axis))
      )

    /** The point with these centre coordinates, without axis indices. */
    def cellBase(cells: Array[Int], cell: Int): Long =
      (0 until n).foldLeft(0L)((key, axis) =>
        key | ((cells(cell * n + axis) + extent).toLong << shift(axis))
      )

    def coordinate(key: Long, axis: Int): Int =
      ((key >>> shift(axis)) & coordinateMask).toInt - extent

    def firstAxis(key: Long): Int = ((key >>> (coordinatesBits + AxisIndexBits)) & axisMask).toInt

    def secondAxis(key: Long): Int = ((key >>> coordinatesBits) & axisMask).toInt

    def centre(key: Long): Array[Int] = Array.tabulate(n)(coordinate(key, _))

    def corner(faceKey: Long, signI: Int, signJ: Int): Long =
      (faceKey & coordinatesMask) +
        signI * unit(firstAxis(faceKey)) + signJ * unit(secondAxis(faceKey))

    private def axisMask = (1L << AxisIndexBits) - 1

  /** One face for every pair of axes and choice of side on the others, as the key to add to a
    * cell's: the sides are +-1 from the cell's centre, a face's half-size is 1. */
  private def faceTemplates(keys: FaceKeys): Array[Long] =
    val n = keys.n
    (for
      i <- 0 until n
      j <- i + 1 until n
      others = (0 until n).filter(axis => axis != i && axis != j)
      sides <- 0 until (1 << others.size)
    yield others.zipWithIndex.foldLeft(keys.header(i, j)) { case (key, (axis, bit)) =>
      key + (if (sides >> bit & 1) == 1 then keys.unit(axis) else -keys.unit(axis))
    }).toArray

  /** The 2-faces of the cube: centre on the cube's boundary in the n-2 axes the face does not
    * span. */
  private def cubeFaces(keys: FaceKeys): Array[Long] =
    val n = keys.n
    val faces = for
      i <- 0 until n
      j <- i + 1 until n
      others = (0 until n).filter(axis => axis != i && axis != j)
      sides <- 0 until (1 << others.size)
    yield
      val centre = new Array[Int](n)
      others.zipWithIndex.foreach { (axis, bit) =>
        centre(axis) = if (sides >> bit & 1) == 1 then keys.extent else -keys.extent
      }
      keys.encode(i, j, centre)
    faces.toArray.sorted

  /** Step `step` of the surface: the 9 parts of every face, the middle one replaced by walls.
    * Each face has a fixed slot for each part it may produce; a part outside the sponge, or a
    * wall that does not exist because the face lies on the other side, leaves its slot empty. */
  private def refine(
    keys: FaceKeys, level: Int, faces: Array[Long], step: Int, k: Int
  ): Array[Long] =
    val n = keys.n
    val half = power(Base, level - step)
    val slots = FlatParts + WallsPerNormal * (n - 2)
    val size = faces.length.toLong * slots
    require(size <= Int.MaxValue, s"step $step of the surface needs $size faces, too many")
    val children = Array.fill(size.toInt)(NoFace)
    faces.indices.foreach { index =>
      partsOf(keys, faces(index), half).foreach { (slot, key) =>
        if inSponge(keys, key, half, level, step, k) then children(index * slots + slot) = key
      }
    }
    Arrays.sort(children)
    uniqueRuns(children, _ => true)

  /** The faces that replace `face`, with their slot among a face's parts. */
  private def partsOf(keys: FaceKeys, face: Long, half: Int): IndexedSeq[(Int, Long)] =
    val i = keys.firstAxis(face)
    val j = keys.secondAxis(face)
    val centre = keys.centre(face)
    val outer = for
      row <- 0 until Base
      column <- 0 until Base
      if row != 1 || column != 1
    yield
      val part = centre.clone
      part(i) += (column - 1) * OuterShift * half
      part(j) += (row - 1) * OuterShift * half
      (row * Base + column, keys.encode(i, j, part))
    val normals = (0 until keys.n).filter(axis => axis != i && axis != j)
    val walls = for
      (normal, normalIndex) <- normals.zipWithIndex
      direction = directionTowardCentre(centre(normal))
      (edgeAxis, acrossAxis, edgeIndex) <- Seq((i, j, 0), (j, i, 1))
      side <- Seq(-1, 1)
    yield
      val wall = centre.clone
      wall(acrossAxis) += side * half
      wall(normal) += direction * half
      val slot = FlatParts + normalIndex * WallsPerNormal +
        edgeIndex * 2 + (if side > 0 then 1 else 0)
      (slot, keys.encode(math.min(edgeAxis, normal), math.max(edgeAxis, normal), wall))
    outer ++ walls

  // Why a surface face never lies on a centre plane: in every surface face the centre
  // coordinates on the two in-plane axes are even and those on the n - 2 normal axes are odd, so
  // a normal coordinate is never 0. By induction over the refinement, in the integer grid of
  // cube extent 3^level (odd):
  // - level 0 (`cubeFaces`): in-plane coordinates are 0, normal ones +-extent, which is odd;
  // - an outer sub-square adds a multiple of OuterShift * half to in-plane coordinates only;
  // - a wall starts from the centre sub-square of its parent (in-plane even, normals odd) and
  //   adds +-half, odd as a power of 3, on the parent's across axis (even becomes odd) and on
  //   the parent's normal (odd becomes even). It spans the edge axis and the old normal, both
  //   even now, and its normals are the across axis (odd now) and the parent's other normals.
  private def directionTowardCentre(coordinate: Int): Int =
    require(coordinate != 0, "a surface face never lies on a centre plane")
    if coordinate > 0 then -1 else 1

  /** A face survives when its centre and its four corners all lie in the closed sponge. */
  private def inSponge(
    keys: FaceKeys, face: Long, half: Int, level: Int, step: Int, k: Int
  ): Boolean =
    val i = keys.firstAxis(face)
    val j = keys.secondAxis(face)
    val centre = keys.centre(face)
    def pointInSponge(signI: Int, signJ: Int): Boolean =
      val point = centre.clone
      point(i) += signI * half
      point(j) += signJ * half
      pointInSpongeAtStep(point, keys.extent, level, step, k)
    pointInSponge(0, 0) &&
      CornerSignsI.indices.forall(c => pointInSponge(CornerSignsI(c), CornerSignsJ(c)))

  /** The base-3 digit test: at every scale 1..step at most k coordinates have the middle digit.
    * A coordinate on a cell boundary has no single digit and never counts as a middle one. */
  private def pointInSpongeAtStep(
    point: Array[Int], extent: Int, level: Int, step: Int, k: Int
  ): Boolean =
    point.forall(math.abs(_) <= extent) &&
      (1 to step).forall { scale =>
        val cellSize = OuterShift * power(Base, level - scale)
        point.count { coordinate =>
          val offset = coordinate + extent
          offset % cellSize != 0 && offset / cellSize % Base == 1
        } <= k
      }

  /** The distinct values of a sorted array whose number of copies passes `keepCount`; the
    * array itself is overwritten. A negative value is no face. */
  private def uniqueRuns(sorted: Array[Long], keepCount: Int => Boolean): Array[Long] =
    @tailrec
    def runEnd(from: Int, next: Int): Int =
      if next < sorted.length && sorted(next) == sorted(from) then runEnd(from, next + 1)
      else next

    @tailrec
    def compact(from: Int, written: Int): Int =
      if from >= sorted.length then written
      else
        val end = runEnd(from, from + 1)
        val keep = sorted(from) >= 0 && keepCount(end - from)
        if keep then sorted(written) = sorted(from)
        compact(end, if keep then written + 1 else written)

    Arrays.copyOf(sorted, compact(0, 0))

  private def meshOf(keys: FaceKeys, faces: Array[Long]): IndexedMeshN =
    meshesBuilt.get.incrementAndGet()
    val n = keys.n
    val cornerKeys = Array.tabulate(faces.length * IndexedMeshN.CornersPerFace) { index =>
      val corner = index % IndexedMeshN.CornersPerFace
      keys.corner(faces(index / IndexedMeshN.CornersPerFace),
        CornerSignsI(corner), CornerSignsJ(corner))
    }
    val sortedCorners = cornerKeys.clone
    Arrays.sort(sortedCorners)
    val vertexKeys = uniqueRuns(sortedCorners, _ => true)
    require(
      vertexKeys.length.toLong * n <= Int.MaxValue,
      s"a ${n}D sponge with ${vertexKeys.length} vertices is too large to build"
    )
    val gridSize = (OuterShift * keys.extent).toFloat
    IndexedMeshN(
      n,
      Array.tabulate(vertexKeys.length * n)(index =>
        keys.coordinate(vertexKeys(index / n), index % n).toFloat / gridSize
      ),
      cornerKeys.map(Arrays.binarySearch(vertexKeys, _))
    )
