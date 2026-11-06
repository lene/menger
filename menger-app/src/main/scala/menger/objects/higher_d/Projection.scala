package menger.objects.higher_d

import scala.annotation.targetName
import scala.math.pow
import scala.math.signum

import menger.common.Vector



case class Projection(eyeW: Float, screenW: Float):
  require(eyeW > 0 && screenW > 0, "eyeW and screenW must be positive")
  require(eyeW > screenW, "eyeW must be greater than screenW")

  final val addExponent = 1.1

  @targetName("plus")
  def +(p: Projection): Projection = Projection(newEyeW(p), screenW)

  private def newEyeW(p: Projection): Float = pow(eyeW, exponent(p)).toFloat
  private def exponent(p: Projection): Double = pow(addExponent, signum(p.eyeW - eyeW))

  
  def factor(depth: Float): Float = (eyeW - screenW) / (eyeW - depth)

  def apply(point: Vector[4]): Vector[3] =
    val projectionFactor = factor(point(3))
    Vector[3](point(0) * projectionFactor, point(1) * projectionFactor, point(2) * projectionFactor)

  
  def apply(points: Seq[Vector[4]]): Seq[Vector[3]] = points.map(apply)

  def apply[V <: Int & Singleton](points: Face4D[V])(using ValueOf[V]): Face3D[V] =
    val vectors: IndexedSeq[Vector[3]] = points.vertices.map(apply)
    Face3D[V](vectors)


// stages.head projects N -> N-1 (dropping the last coordinate), stages.last projects 4 -> 3;
// each stage's eyeW / screenW are the eye and screen along the coordinate it drops.
case class PerspectiveChain[N <: Int & Singleton](stages: Seq[Projection])(using d: ValueOf[N]):
  Dimension.requireSupported(d.value, "perspective chain")
  require(
    stages.size == d.value - PerspectiveChain.targetDimension,
    s"a perspective chain from ${d.value}D to ${PerspectiveChain.targetDimension}D needs " +
      s"${d.value - PerspectiveChain.targetDimension} stages, got ${stages.size}"
  )

  // Does not clip: a point at or behind a stage's eye gives Inf/NaN or a mirrored point.
  // Use `project` for points that can reach a stage's eye.
  def apply(point: Vector[N]): Vector[3] =
    toVector3(stages.foldLeft(point.toIndexedSeq)(PerspectiveChain.dropLast))

  def project(point: Vector[N]): Option[Vector[3]] =
    stages.foldLeft(Option(point.toIndexedSeq)) { (coordinates, stage) =>
      coordinates
        .filterNot(c => PerspectiveChain.isClipped(c, stage))
        .map(PerspectiveChain.dropLast(_, stage))
    }.map(toVector3)

  private def toVector3(coordinates: IndexedSeq[Float]): Vector[3] = Vector[3](coordinates*)

object PerspectiveChain:
  val targetDimension: Int = 3
  // matches the epsilon the 4D CUDA shaders use for the eye_w clip (e.g. hit_menger4d.cu)
  val clipEpsilon: Float = 1e-6f

  def uniform[N <: Int & Singleton](eye: Float, screen: Float)(
    using d: ValueOf[N]
  ): PerspectiveChain[N] =
    PerspectiveChain[N](Seq.fill(d.value - targetDimension)(Projection(eye, screen)))

  private def dropLast(coordinates: IndexedSeq[Float], stage: Projection): IndexedSeq[Float] =
    val factor = stage.factor(coordinates.last)
    coordinates.init.map(_ * factor)

  private def isClipped(coordinates: IndexedSeq[Float], stage: Projection): Boolean =
    coordinates.last >= stage.eyeW - clipEpsilon
