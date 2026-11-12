package menger.objects.higher_d

import java.util.Arrays

import scala.annotation.tailrec

enum NdSpongeVariant:
  case Volume, Surface

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
  // per normal axis: both directions, the face's two edge axes, both sides of the hole
  private val WallsPerNormal = 8
  private val OuterShift = 2
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
    requireBuildable(
      saturatingPower(keptPerStep(n, k), level), Int.MaxValue / n, n, level, k, "kept cells"
    )
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
    val finest = (1 to level).foldLeft(cubeFaces(keys)) { (faces, step) =>
      refine(keys, level, faces, step, k)
    }
    meshOf(keys, finest)

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

  /** What `refine` reserves for one face: the 8 outer squares of the 3 x 3 split, and for each
    * of the n - 2 normals up to 8 walls (both directions, 2 edge axes, 2 sides) for the middle
    * square. The surface starts from C(n,2) 2^(n-2) faces and multiplies by this once per level,
    * so no step has more faces than this bound allows. */
  private def surfaceChildrenPerFace(n: Int): Long =
    FlatParts - 1L + WallsPerNormal.toLong * (n - 2)

  private def requireBuildableFaces(variant: NdSpongeVariant, n: Int, level: Int, k: Int): Unit =
    requireBuildable(
      upperBoundFaces(variant, n, level, k), Int.MaxValue / IndexedMeshN.CornersPerFace,
      n, level, k, s"$variant faces"
    )

  private def requireBuildable(
    candidates: Long, limit: Int, n: Int, level: Int, k: Int, what: String
  ): Unit =
    require(
      candidates <= limit,
      s"a ${n}D sponge of level $level (k=$k) is too large to build: up to $candidates $what, " +
        s"more than an array can hold ($limit)"
    )

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

  private def requireKeyFits(n: Int, level: Int, k: Int): Unit =
    requireDimensionAndRange(n, k)
    require(level >= 0, s"level must not be negative, got $level")
    val bits = coordinateBits(level)
    // ponytail: n * bits + 6 <= 63 caps the level at 3 for n = 8, 4 for n = 7, 5 for n = 6,
    // 6 for n = 5 and 8 for n = 4; wider keys would need two Longs per face.
    require(
      n * bits + 2 * AxisIndexBits <= KeyBitBudget,
      s"N=$n, L=$level: the face key is too wide ($n coordinates of $bits bits and two axis " +
        s"indices do not fit in $KeyBitBudget bits)"
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
      direction <- directionsTowardCentre(centre(normal))
      (edgeAxis, acrossAxis, edgeIndex) <- Seq((i, j, 0), (j, i, 1))
      side <- Seq(-1, 1)
    yield
      val wall = centre.clone
      wall(acrossAxis) += side * half
      wall(normal) += direction * half
      val slot = FlatParts + normalIndex * WallsPerNormal +
        (if direction > 0 then 4 else 0) + edgeIndex * 2 + (if side > 0 then 1 else 0)
      (slot, keys.encode(math.min(edgeAxis, normal), math.max(edgeAxis, normal), wall))
    outer ++ walls

  private def directionsTowardCentre(coordinate: Int): Seq[Int] =
    if coordinate > 0 then Seq(-1) else if coordinate < 0 then Seq(1) else Seq(-1, 1)

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
