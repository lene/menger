package menger.engines.scene

import scala.annotation.tailrec

import menger.CurveData
import menger.ObjectSpec
import menger.common.Color
import menger.common.Material
import menger.common.Vector
import menger.dsl.Vec3
import menger.objects.LSystemGrammar
import menger.objects.LSystemPresets
import menger.objects.higher_d.Projection
import menger.objects.higher_d.Rotation

private type SVec[+A] = scala.Vector[A]
private val SVec = scala.Vector

private case class TurtleState4D(
  pos: Vector[4],
  heading: Vector[4],
  left: Vector[4],
  up: Vector[4],
  ana: Vector[4],
  width: Float
)

private case class StepResult4D(
  nextI: Int,
  state: TurtleState4D,
  specs: List[ObjectSpec],
  runPoints: SVec[Vector[4]],
  runWidths: SVec[Float],
  stack: List[TurtleState4D],
  skipCount: Int
)

class LSystemTurtle4D(
  grammarString: String,
  angleDegrees: Float,
  segmentLength: Float,
  initialWidth: Float = 0.1f,
  widthDecay: Float = 0.7f,
  seed: Long = 42L,
  rotXW: Float = 15f,
  rotYW: Float = 10f,
  rotZW: Float = 0f,
  eyeW: Float = 3.0f,
  screenW: Float = 1.5f
):

  import LSystemTurtle4D.{DegToRad, rotateVector4D, project4DTo3D}

  private val angleRad: Float = angleDegrees * DegToRad
  private val rotation = Rotation(
    degreesXW = rotXW, degreesYW = rotYW, degreesZW = rotZW, pivotPoint = Vector.Zero[4]
  )
  private val projection = Projection(eyeW, screenW)

  private val headingIdx = 1
  private val leftIdx = 0
  private val upIdx = 2
  private val anaIdx = 3

  private val defaultMat: Material = Material(Color(0.7f, 0.7f, 0.7f))

  private val initHeading: Vector[4] = Vector.Y
  private val initLeft: Vector[4] = Vector.X
  private val initUp: Vector[4] = Vector.Z
  private val initAna: Vector[4] = Vector.W

  private val initialState: TurtleState4D = TurtleState4D(
    Vector.Zero[4], initHeading, initLeft, initUp, initAna, initialWidth
  )

  def generate(): List[ObjectSpec] =
    process(grammarString, 0, initialState, List.empty,
      SVec.empty, SVec.empty, List.empty, 0)

  @tailrec
  private def process(
    s: String,
    i: Int,
    state: TurtleState4D,
    specs: List[ObjectSpec],
    runPoints: SVec[Vector[4]],
    runWidths: SVec[Float],
    stack: List[TurtleState4D],
    skipCount: Int
  ): List[ObjectSpec] =
    if i >= s.length then
      emitRun(specs, runPoints, runWidths)._1
    else if skipCount > 0 then
      process(s, i + 1, state, specs, runPoints, runWidths, stack, skipCount - 1)
    else
      val result = stepSymbol(s, i, state, specs, runPoints, runWidths, stack)
      process(s, result.nextI, result.state, result.specs,
        result.runPoints, result.runWidths, result.stack, result.skipCount)

  private def stepSymbol(
    s: String, i: Int, state: TurtleState4D, specs: List[ObjectSpec],
    runPoints: SVec[Vector[4]], runWidths: SVec[Float],
    stack: List[TurtleState4D]
  ): StepResult4D =
    s(i) match
      case 'F' => stepF(state, specs, runPoints, runWidths, stack, i)
      case 'f' => stepFwdNoRecord(state, specs, runPoints, runWidths, stack, i)
      case '+' => stepTurn(state, specs, runPoints, runWidths, stack, i,
          headingIdx, leftIdx, angleRad)
      case '-' => stepTurn(state, specs, runPoints, runWidths, stack, i,
          headingIdx, leftIdx, -angleRad)
      case '&' => stepTurn(state, specs, runPoints, runWidths, stack, i,
          headingIdx, upIdx, angleRad)
      case '^' => stepTurn(state, specs, runPoints, runWidths, stack, i,
          headingIdx, upIdx, -angleRad)
      case '\\' => stepTurn(state, specs, runPoints, runWidths, stack, i,
          leftIdx, upIdx, angleRad)
      case '/' => stepTurn(state, specs, runPoints, runWidths, stack, i,
          leftIdx, upIdx, -angleRad)
      case '>' => stepTurn(state, specs, runPoints, runWidths, stack, i,
          headingIdx, anaIdx, angleRad)
      case '<' => stepTurn(state, specs, runPoints, runWidths, stack, i,
          headingIdx, anaIdx, -angleRad)
      case '|' => stepTurn180(state, specs, runPoints, runWidths, stack, i)
      case '[' => stepPush(state, specs, runPoints, runWidths, stack, i)
      case ']' => stepPop(state, specs, runPoints, runWidths, stack, i)
      case '!' => stepWidth(state, specs, runPoints, runWidths, stack, i)
      case _ => stepSkip(state, specs, runPoints, runWidths, stack, i)

  private def stepF(
    state: TurtleState4D, specs: List[ObjectSpec],
    runPoints: SVec[Vector[4]], runWidths: SVec[Float],
    stack: List[TurtleState4D], i: Int
  ): StepResult4D =
    val newPos = state.pos + state.heading * segmentLength
    val newState = state.copy(pos = newPos)
    StepResult4D(i + 1, newState, specs,
      runPoints :+ newPos, runWidths :+ state.width, stack, 0)

  private def stepFwdNoRecord(
    state: TurtleState4D, specs: List[ObjectSpec],
    runPoints: SVec[Vector[4]], runWidths: SVec[Float],
    stack: List[TurtleState4D], i: Int
  ): StepResult4D =
    val (newSpecs, _) = emitRun(specs, runPoints, runWidths)
    val newPos = state.pos + state.heading * segmentLength
    StepResult4D(i + 1, state.copy(pos = newPos), newSpecs,
      SVec.empty, SVec.empty, stack, 0)

  private def stepTurn(
    state: TurtleState4D, specs: List[ObjectSpec],
    runPoints: SVec[Vector[4]], runWidths: SVec[Float],
    stack: List[TurtleState4D], i: Int,
    planeI: Int, planeJ: Int, angle: Float
  ): StepResult4D =
    val newHeading = rotateVector4D(state.heading, planeI, planeJ, angle)
    val newLeft = rotateVector4D(state.left, planeI, planeJ, angle)
    val newUp = rotateVector4D(state.up, planeI, planeJ, angle)
    val newAna = rotateVector4D(state.ana, planeI, planeJ, angle)
    StepResult4D(i + 1, state.copy(
      heading = newHeading, left = newLeft, up = newUp, ana = newAna),
      specs, runPoints, runWidths, stack, 0)

  private def stepTurn180(
    state: TurtleState4D, specs: List[ObjectSpec],
    runPoints: SVec[Vector[4]], runWidths: SVec[Float],
    stack: List[TurtleState4D], i: Int
  ): StepResult4D =
    StepResult4D(i + 1, state.copy(
      heading = -state.heading, left = -state.left),
      specs, runPoints, runWidths, stack, 0)

  private def stepPush(
    state: TurtleState4D, specs: List[ObjectSpec],
    runPoints: SVec[Vector[4]], runWidths: SVec[Float],
    stack: List[TurtleState4D], i: Int
  ): StepResult4D =
    StepResult4D(i + 1, state, specs, runPoints, runWidths,
      state :: stack, 0)

  private def stepPop(
    state: TurtleState4D, specs: List[ObjectSpec],
    runPoints: SVec[Vector[4]], runWidths: SVec[Float],
    stack: List[TurtleState4D], i: Int
  ): StepResult4D =
    val (newSpecs, _) = emitRun(specs, runPoints, runWidths)
    stack match
      case popped :: rest =>
        StepResult4D(i + 1, popped, newSpecs, SVec.empty, SVec.empty, rest, 0)
      case Nil =>
        StepResult4D(i + 1, state, newSpecs, SVec.empty, SVec.empty, Nil, 0)

  private def stepWidth(
    state: TurtleState4D, specs: List[ObjectSpec],
    runPoints: SVec[Vector[4]], runWidths: SVec[Float],
    stack: List[TurtleState4D], i: Int
  ): StepResult4D =
    StepResult4D(i + 1, state.copy(width = state.width * widthDecay),
      specs, runPoints, runWidths, stack, 0)

  private def stepSkip(
    state: TurtleState4D, specs: List[ObjectSpec],
    runPoints: SVec[Vector[4]], runWidths: SVec[Float],
    stack: List[TurtleState4D], i: Int
  ): StepResult4D =
    StepResult4D(i + 1, state, specs, runPoints, runWidths, stack, 0)

  private def emitRun(
    specs: List[ObjectSpec],
    points: SVec[Vector[4]],
    widths: SVec[Float]
  ): (List[ObjectSpec], (SVec[Vector[4]], SVec[Float])) =
    if points.isEmpty then (specs, (SVec.empty, SVec.empty))
    else
      val projectedPoints = points.map(v4 => project4DTo3D(v4, rotation, projection))
      val flatPoints = SVec.from(projectedPoints.flatMap(p => Seq(p.x, p.y, p.z)))
      val flatWidths = SVec.from(widths)
      val spec = ObjectSpec(
        objectType = "curve",
        curveData = Some(CurveData(flatPoints, flatWidths)),
        material = Some(defaultMat)
      )
      (spec :: specs, (SVec.empty, SVec.empty))

object LSystemTurtle4D:

  val DegToRad: Float = (math.Pi / 180.0).toFloat
  val DefaultMat: Material = Material(Color(0.7f, 0.7f, 0.7f))

  def rotateVector4D(v: Vector[4], i: Int, j: Int, angleRad: Float): Vector[4] =
    val c = math.cos(angleRad).toFloat
    val s = math.sin(angleRad).toFloat
    val vi = v(i)
    val vj = v(j)
    Vector[4](
      if 0 == i then c * vi - s * vj else if 0 == j then s * vi + c * vj else v(0),
      if 1 == i then c * vi - s * vj else if 1 == j then s * vi + c * vj else v(1),
      if 2 == i then c * vi - s * vj else if 2 == j then s * vi + c * vj else v(2),
      if 3 == i then c * vi - s * vj else if 3 == j then s * vi + c * vj else v(3)
    )

  def project4DTo3D(v: Vector[4], rot: Rotation, proj: Projection): Vec3 =
    val rotated = rot(v)
    val gdx3 = proj(rotated)
    Vec3(gdx3.x, gdx3.y, gdx3.z)

  private def grammar(axiom: String, rules: Map[Char, Seq[(Double, String)]],
    iterations: Int, seed: Long = 42L): String =
    LSystemGrammar(axiom, rules, seed).rewrite(iterations)

  /** 4D Hilbert curve — delegates to LSystemPresets (A5, Sprint 32). */
  val HilbertCurve4D: LSystemTurtle4D = fromPreset("hilbert4d", 90f, 0.15f, 0.05f, 0.7f)

  /** 4D tree — delegates to LSystemPresets (A5, Sprint 32). */
  val Tree4D: LSystemTurtle4D = fromPreset("tree4d", 25.7f, 0.3f, 0.08f, 0.7f)

  private def fromPreset(name: String, angle: Float, segLen: Float, initWidth: Float, decay: Float): LSystemTurtle4D =
    val (axiom, rules, _, _, _, _, _) = LSystemPresets(name)
    val g = grammar(axiom, rules.view.mapValues(v => Seq((1.0, v))).toMap, 4)
    LSystemTurtle4D(g, angle, segLen, initWidth, decay)
