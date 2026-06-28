package menger.objects

import scala.annotation.tailrec
import scala.util.Try

import menger.CurveData
import menger.ObjectSpec
import menger.common.Color
import menger.common.Material
import menger.dsl.Vec3

private case class TurtleState3D(
  pos: Vec3,
  heading: Vec3,
  left: Vec3,
  up: Vec3,
  width: Float,
  materialIndex: Int,
  currentTexture: Option[String] = None
)

private case class StepResult(
  nextI: Int,
  state: TurtleState3D,
  specs: List[ObjectSpec],
  runPoints: Vector[Vec3],
  runWidths: Vector[Float],
  stack: List[TurtleState3D],
  skipCount: Int
)

class LSystemTurtle3D(
  grammarString: String,
  angleDegrees: Float,
  segmentLength: Float,
  initialWidth: Float = 0.1f,
  widthDecay: Float = 0.7f,
  seed: Long = 42L,
  materials: Map[String, Material] = Map.empty,
  defaultMaterial: Material = LSystemTurtle3D.DefaultMat,
  normalizeScale: Boolean = true,
  curvesEnabled: Boolean = true
):

  import LSystemTurtle3D.{DegToRad, rotateAboutAxis}

  private val angleRad: Float = angleDegrees * DegToRad

  private val initialHeading: Vec3 = Vec3.UnitY
  private val initialLeft: Vec3 = Vec3.UnitX
  private val initialUp: Vec3 = Vec3.UnitZ
  private val initialState: TurtleState3D = TurtleState3D(
    Vec3.Zero, initialHeading, initialLeft, initialUp, initialWidth, 0
  )

  private val materialList: Vector[Material] =
    if materials.nonEmpty then materials.values.toVector
    else Vector(defaultMaterial)

  private val materialNameToIndex: Map[String, Int] =
    if materials.nonEmpty then
      materials.keys.zipWithIndex.toMap
    else
      Map.empty

  def generate(): List[ObjectSpec] =
    val rawSpecs = process(grammarString, 0, initialState, List.empty,
      Vector.empty, Vector.empty, List.empty, 0)
    if normalizeScale then normalize(rawSpecs) else rawSpecs

  @tailrec
  private def process(
    s: String,
    i: Int,
    state: TurtleState3D,
    specs: List[ObjectSpec],
    runPoints: Vector[Vec3],
    runWidths: Vector[Float],
    stack: List[TurtleState3D],
    skipCount: Int
  ): List[ObjectSpec] =
    if i >= s.length then
      emitRun(state, specs, runPoints, runWidths)._1
    else if skipCount > 0 then
      process(s, i + 1, state, specs, runPoints, runWidths, stack, skipCount - 1)
    else
      val result = stepSymbol(s, i, state, specs, runPoints, runWidths, stack)
      process(s, result.nextI, result.state, result.specs,
        result.runPoints, result.runWidths, result.stack, result.skipCount)

  private def stepSymbol(
    s: String, i: Int, state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float], stack: List[TurtleState3D]
  ): StepResult =
    s(i) match
      case 'F' => stepF(s, i, state, specs, runPoints, runWidths, stack)
      case 'f' => stepFwdNoRecord(state, specs, runPoints, runWidths, stack, i)
      case '+' => stepTurn(state, specs, runPoints, runWidths, stack, i, state.up, angleRad)
      case '-' => stepTurn(state, specs, runPoints, runWidths, stack, i, state.up, -angleRad)
      case '&' => stepTurn(state, specs, runPoints, runWidths, stack, i, state.left, angleRad)
      case '^' => stepTurn(state, specs, runPoints, runWidths, stack, i, state.left, -angleRad)
      case '\\' => stepTurn(state, specs, runPoints, runWidths, stack, i, state.heading, angleRad)
      case '/' => stepTurn(state, specs, runPoints, runWidths, stack, i, state.heading, -angleRad)
      case '<' => stepTurn(state, specs, runPoints, runWidths, stack, i, state.up, angleRad)
      case '>' => stepTurn(state, specs, runPoints, runWidths, stack, i, state.up, -angleRad)
      case '|' => stepTurn180(state, specs, runPoints, runWidths, stack, i)
      case '[' => stepPush(state, specs, runPoints, runWidths, stack, i)
      case ']' => stepPop(state, specs, runPoints, runWidths, stack, i)
      case '!' => stepWidth(s, i, state, specs, runPoints, runWidths, stack)
      case '\'' => stepIncMaterial(state, specs, runPoints, runWidths, stack, i)
      case '"' => stepDecMaterial(state, specs, runPoints, runWidths, stack, i)
      case '%' => stepPrune(s, i, state, specs, runPoints, runWidths, stack)
      case '@' => stepAt(s, i, state, specs, runPoints, runWidths, stack)
      case 'J' => stepMeshStamp(s, i, state, specs, runPoints, runWidths, stack)
      case '{' | '}' => stepBrace(state, specs, runPoints, runWidths, stack, i)
      case 'M' => stepMaterial(s, i, state, specs, runPoints, runWidths, stack)
      case 'T' => stepTexture(s, i, state, specs, runPoints, runWidths, stack)
      case _ => stepSkip(state, specs, runPoints, runWidths, stack, i)

  private def stepF(
    s: String, i: Int, state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float], stack: List[TurtleState3D]
  ): StepResult =
    val (segLen, segWidth, shape, nextIdx) = parseFParams(s, i, state.width)
    val newPos = state.pos + state.heading * segLen
    val newState = state.copy(pos = newPos, width = segWidth)
    shape match
      case "sphere" =>
        val (newSpecs, _) = emitRun(state, specs, runPoints, runWidths)
        val mat = currentMaterial(state)
        val sphereSpec = ObjectSpec(
          objectType = "sphere", x = newPos.x, y = newPos.y, z = newPos.z,
          size = segWidth * 2f, material = Some(mat),
          texture = state.currentTexture
        )
        StepResult(nextIdx, newState, sphereSpec :: newSpecs,
          Vector.empty, Vector.empty, stack, 0)
      case _ =>
        StepResult(nextIdx, newState, specs,
          runPoints :+ newPos, runWidths :+ segWidth, stack, 0)

  private def stepFwdNoRecord(
    state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float],
    stack: List[TurtleState3D], i: Int
  ): StepResult =
    val (newSpecs, _) = emitRun(state, specs, runPoints, runWidths)
    val newPos = state.pos + state.heading * segmentLength
    StepResult(i + 1, state.copy(pos = newPos), newSpecs,
      Vector.empty, Vector.empty, stack, 0)

  private def stepTurn(
    state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float],
    stack: List[TurtleState3D], i: Int, axis: Vec3, angle: Float
  ): StepResult =
    val (newSpecs, _) = emitRun(state, specs, runPoints, runWidths)
    val newHeading = rotateAboutAxis(state.heading, axis, angle)
    val newLeft = rotateAboutAxis(state.left, axis, angle)
    val newUp = rotateAboutAxis(state.up, axis, angle)
    StepResult(i + 1, state.copy(heading = newHeading, left = newLeft, up = newUp),
      newSpecs, Vector.empty, Vector.empty, stack, 0)

  private def stepTurn180(
    state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float],
    stack: List[TurtleState3D], i: Int
  ): StepResult =
    val (newSpecs, _) = emitRun(state, specs, runPoints, runWidths)
    val cosPi = math.cos(math.Pi).toFloat
    val sinPi = math.sin(math.Pi).toFloat
    val axis = state.up
    val newHeading = state.heading * cosPi + axis.cross(state.heading) * sinPi +
      axis * (axis.dot(state.heading) * (1f - cosPi))
    val newLeft = state.left * cosPi + axis.cross(state.left) * sinPi +
      axis * (axis.dot(state.left) * (1f - cosPi))
    StepResult(i + 1, state.copy(heading = newHeading, left = newLeft),
      newSpecs, Vector.empty, Vector.empty, stack, 0)

  private def stepPush(
    state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float],
    stack: List[TurtleState3D], i: Int
  ): StepResult =
    val (newSpecs, _) = emitRun(state, specs, runPoints, runWidths)
    StepResult(i + 1, state, newSpecs, Vector.empty, Vector.empty,
      state :: stack, 0)

  private def stepPop(
    state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float],
    stack: List[TurtleState3D], i: Int
  ): StepResult =
    val (newSpecs, _) = emitRun(state, specs, runPoints, runWidths)
    stack match
      case popped :: rest =>
        StepResult(i + 1, popped, newSpecs, Vector.empty, Vector.empty, rest, 0)
      case Nil =>
        StepResult(i + 1, state, newSpecs, Vector.empty, Vector.empty, Nil, 0)

  private def stepIncMaterial(
    state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float],
    stack: List[TurtleState3D], i: Int
  ): StepResult =
    val ni = (state.materialIndex + 1) % math.max(1, materialList.length)
    StepResult(i + 1, state.copy(materialIndex = ni),
      specs, runPoints, runWidths, stack, 0)

  private def stepDecMaterial(
    state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float],
    stack: List[TurtleState3D], i: Int
  ): StepResult =
    val ni = (state.materialIndex - 1 + materialList.length) % math.max(1, materialList.length)
    StepResult(i + 1, state.copy(materialIndex = ni),
      specs, runPoints, runWidths, stack, 0)

  private def stepWidth(
    s: String, i: Int, state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float], stack: List[TurtleState3D]
  ): StepResult =
    if i + 1 < s.length && s(i + 1) == '(' then
      val (args, nextIdx) = parseParenArgs(s, i + 2)
      val factor = args.headOption.flatMap(a => Try(a.toFloat).toOption).getOrElse(1f)
      StepResult(nextIdx, state.copy(width = state.width * factor),
        specs, runPoints, runWidths, stack, 0)
    else
      StepResult(i + 1, state.copy(width = state.width * widthDecay),
        specs, runPoints, runWidths, stack, 0)

  private def stepPrune(
    s: String, i: Int, state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float], stack: List[TurtleState3D]
  ): StepResult =
    if i + 1 < s.length && s(i + 1) == '(' then
      val (args, nextIdx) = parseParenArgs(s, i + 2)
      val count = args.headOption.flatMap(a => Try(a.toInt).toOption).getOrElse(0)
      StepResult(nextIdx, state, specs, runPoints, runWidths, stack, count)
    else
      StepResult(i + 1, state, specs, runPoints, runWidths, stack, 0)

  private def stepAt(
    s: String, i: Int, state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float], stack: List[TurtleState3D]
  ): StepResult =
    if i + 1 < s.length then
      s(i + 1) match
        case 'O' => stepSpherePrim(s, i + 1, state, specs, runPoints, runWidths, stack)
        case 'c' => stepDiskPrim(s, i + 1, state, specs, runPoints, runWidths, stack)
        case _ => stepSkip(state, specs, runPoints, runWidths, stack, i + 1)
    else
      stepSkip(state, specs, runPoints, runWidths, stack, i)

  private def stepSpherePrim(
    s: String, idx: Int, state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float], stack: List[TurtleState3D]
  ): StepResult =
    if idx + 1 < s.length && s(idx + 1) == '(' then
      val (args, nextIdx) = parseParenArgs(s, idx + 2)
      val dia = args.headOption.flatMap(a => Try(a.toFloat).toOption).getOrElse(1f)
      val mat = currentMaterial(state)
      val sphereSpec = ObjectSpec(
        objectType = "sphere", x = state.pos.x, y = state.pos.y, z = state.pos.z,
        size = dia, material = Some(mat),
        texture = state.currentTexture
      )
      StepResult(nextIdx, state, sphereSpec :: specs,
        runPoints, runWidths, stack, 0)
    else
      StepResult(idx + 2, state, specs, runPoints, runWidths, stack, 0)

  private def stepDiskPrim(
    s: String, idx: Int, state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float], stack: List[TurtleState3D]
  ): StepResult =
    if idx + 1 < s.length && s(idx + 1) == '(' then
      val (args, nextIdx) = parseParenArgs(s, idx + 2)
      val dia = args.headOption.flatMap(a => Try(a.toFloat).toOption).getOrElse(1f)
      val mat = currentMaterial(state)
      val diskSpec = makeDiskSpec(state.pos, state.up, dia, mat, state.currentTexture)
      StepResult(nextIdx, state, diskSpec :: specs,
        runPoints, runWidths, stack, 0)
    else
      StepResult(idx + 2, state, specs, runPoints, runWidths, stack, 0)

  private def stepMeshStamp(
    s: String, i: Int, state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float], stack: List[TurtleState3D]
  ): StepResult =
    if i + 1 < s.length && s(i + 1) == '(' then
      val (args, nextIdx) = parseParenArgs(s, i + 2)
      args match
        case name :: scaleStr :: _ =>
          val scale = Try(scaleStr.toFloat).getOrElse(1f)
          val cleanName = name.stripPrefix("\"").stripSuffix("\"")
          val mat = currentMaterial(state)
          val stampSpec = ObjectSpec(
            objectType = cleanName, x = state.pos.x, y = state.pos.y, z = state.pos.z,
            size = scale, material = Some(mat),
            texture = state.currentTexture
          )
          StepResult(nextIdx, state, stampSpec :: specs,
            runPoints, runWidths, stack, 0)
        case _ =>
          StepResult(nextIdx, state, specs, runPoints, runWidths, stack, 0)
    else
      StepResult(i + 1, state, specs, runPoints, runWidths, stack, 0)

  private def stepMaterial(
    s: String, i: Int, state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float], stack: List[TurtleState3D]
  ): StepResult =
    if i + 1 < s.length && s(i + 1) == '(' then
      val (args, nextIdx) = parseParenArgs(s, i + 2)
      val name = args.headOption.getOrElse("")
      val idx = materialNameToIndex.getOrElse(name, state.materialIndex)
      StepResult(nextIdx, state.copy(materialIndex = idx),
        specs, runPoints, runWidths, stack, 0)
    else
      StepResult(i + 1, state, specs, runPoints, runWidths, stack, 0)

  private def stepTexture(
    s: String, i: Int, state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float], stack: List[TurtleState3D]
  ): StepResult =
    if i + 1 < s.length && s(i + 1) == '(' then
      val (args, nextIdx) = parseParenArgs(s, i + 2)
      val filename = args.headOption
        .map(f => f.stripPrefix("\"").stripSuffix("\""))
      StepResult(nextIdx, state.copy(currentTexture = filename),
        specs, runPoints, runWidths, stack, 0)
    else
      StepResult(i + 1, state, specs, runPoints, runWidths, stack, 0)

  private def stepBrace(
    state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float],
    stack: List[TurtleState3D], i: Int
  ): StepResult =
    StepResult(i + 1, state, specs, runPoints, runWidths, stack, 0)

  private def stepSkip(
    state: TurtleState3D, specs: List[ObjectSpec],
    runPoints: Vector[Vec3], runWidths: Vector[Float],
    stack: List[TurtleState3D], i: Int
  ): StepResult =
    StepResult(i + 1, state, specs, runPoints, runWidths, stack, 0)

  private def emitRun(
    state: TurtleState3D, specs: List[ObjectSpec],
    points: Vector[Vec3], widths: Vector[Float]
  ): (List[ObjectSpec], (Vector[Vec3], Vector[Float])) =
    if points.isEmpty then (specs, (Vector.empty, Vector.empty))
    else
      val mat = currentMaterial(state)
      val flatPoints = Vector.from(points.flatMap(p => Seq(p.x, p.y, p.z)))
      val flatWidths = Vector.from(widths)
      val spec = ObjectSpec(
        objectType = "curve",
        curveData = Some(CurveData(flatPoints, flatWidths)),
        material = Some(mat),
        texture = state.currentTexture
      )
      (spec :: specs, (Vector.empty, Vector.empty))

  private def makeDiskSpec(pos: Vec3, up: Vec3, diameter: Float, mat: Material,
    texture: Option[String] = None): ObjectSpec =
    val halfHeight = 0.001f * diameter
    val apexPt = (pos.x + up.x * halfHeight, pos.y + up.y * halfHeight, pos.z + up.z * halfHeight)
    val basePt = (pos.x - up.x * halfHeight, pos.y - up.y * halfHeight, pos.z - up.z * halfHeight)
    ObjectSpec(
      objectType = "cone",
      x = pos.x, y = pos.y, z = pos.z,
      size = diameter,
      cone = menger.ConeGeometry(
        apex = Some(apexPt),
        base = Some(basePt),
        radius = Some(diameter / 2f)
      ),
      material = Some(mat),
      texture = texture
    )

  private def parseFParams(s: String, i: Int, currentWidth: Float): (Float, Float, String, Int) =
    if i + 1 < s.length && s(i + 1) == '(' then
      val (args, nextIdx) = parseParenArgs(s, i + 2)
      args.length match
        case 0 => (segmentLength, currentWidth, "cylinder", nextIdx)
        case 1 =>
          val len = Try(args(0).toFloat).getOrElse(segmentLength)
          (len, currentWidth, "cylinder", nextIdx)
        case 2 =>
          val len = Try(args(0).toFloat).getOrElse(segmentLength)
          val w = Try(args(1).toFloat).getOrElse(currentWidth)
          (len, w, "cylinder", nextIdx)
        case _ =>
          val len = Try(args(0).toFloat).getOrElse(segmentLength)
          val w = Try(args(1).toFloat).getOrElse(currentWidth)
          val shape = args(2).stripPrefix("\"").stripSuffix("\"").toLowerCase
          (len, w, shape, nextIdx)
    else
      (segmentLength, currentWidth, "cylinder", i + 1)

  private def parseParenArgs(s: String, start: Int): (List[String], Int) =
    val (content, endIdx) = collectParenContent(s, start, 1, new StringBuilder())
    val args = if content.trim.isEmpty then List.empty[String]
    else splitArgs(content)
    (args, endIdx)

  private def collectParenContent(
    s: String, i: Int, depth: Int, buf: StringBuilder
  ): (String, Int) =
    if i >= s.length || depth == 0 then (buf.toString(), i)
    else
      s(i) match
        case '(' => collectParenContent(s, i + 1, depth + 1, buf.append('('))
        case ')' =>
          if depth == 1 then collectParenContent(s, i + 1, 0, buf)
          else collectParenContent(s, i + 1, depth - 1, buf.append(')'))
        case c => collectParenContent(s, i + 1, depth, buf.append(c))

  private def splitArgs(content: String): List[String] =
    val (parts, _) = splitArgsRec(content, 0, false, new StringBuilder(), List.empty)
    parts

  private def splitArgsRec(
    s: String, i: Int, inQuotes: Boolean,
    current: StringBuilder, acc: List[String]
  ): (List[String], StringBuilder) =
    if i >= s.length then
      val finalAcc = if current.nonEmpty then (current.toString().trim :: acc) else acc
      (finalAcc.reverse, current)
    else
      s(i) match
        case '"' =>
          splitArgsRec(s, i + 1, !inQuotes, current.append('"'), acc)
        case ',' if !inQuotes =>
          splitArgsRec(s, i + 1, false, new StringBuilder(), current.toString().trim :: acc)
        case c =>
          splitArgsRec(s, i + 1, inQuotes, current.append(c), acc)

  private def currentMaterial(state: TurtleState3D): Material =
    if materialList.isEmpty then defaultMaterial
    else
      val idx = state.materialIndex % materialList.length
      materialList(math.abs(idx))

  private def normalize(specs: List[ObjectSpec]): List[ObjectSpec] =
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

object LSystemTurtle3D:

  val DegToRad: Float = (math.Pi / 180.0).toFloat
  val DefaultMat: Material = Material(Color(0.7f, 0.7f, 0.7f))

  def rotateAboutAxis(v: Vec3, axis: Vec3, angleRad: Float): Vec3 =
    val c = math.cos(angleRad).toFloat
    val sVal = math.sin(angleRad).toFloat
    v * c + axis.cross(v) * sVal + axis * (axis.dot(v) * (1f - c))

  private def grammar(axiom: String, rules: Map[Char, Seq[(Double, String)]],
    iterations: Int, seed: Long = 42L): String =
    LSystemGrammar(axiom, rules, seed).rewrite(iterations)

  val Tree: LSystemTurtle3D =
    val g = grammar("F", Map('F' -> Seq((1.0, "F[+F]F[-F]F"))), 4)
    LSystemTurtle3D(g, 25.7f, 0.4f, 0.08f, 0.7f)

  val Bush: LSystemTurtle3D =
    val g = grammar("F", Map('F' -> Seq((1.0, "FF+[+F-F-F]-[-F+F+F]"))), 3)
    LSystemTurtle3D(g, 22.5f, 0.3f, 0.06f, 0.8f)

  val Fern3D: LSystemTurtle3D =
    val g = grammar("F", Map('F' -> Seq((1.0, "F[&+F]F[^&-F][&+F][^F]F"))), 4)
    LSystemTurtle3D(g, 25.7f, 0.4f, 0.08f, 0.7f)

  val HilbertCurve3D: LSystemTurtle3D =
    val g = grammar("X",
      Map(
        'X' -> Seq((1.0, "&F^<XFX-F^>>XFX&F+>>XFX-F>X->")),
        'F' -> Seq((1.0, "F"))
      ), 4)
    LSystemTurtle3D(g, 90f, 0.15f, 0.04f, 0.7f)

  val KochIsland: LSystemTurtle3D =
    val g = grammar("F+F+F+F",
      Map('F' -> Seq((1.0, "F+f-FF+F+FF+Ff+FF-f+FF-F-FF-Ff-FFF"))), 2)
    LSystemTurtle3D(g, 90f, 0.6f, 0.04f, 0.7f)
