package menger

import com.typesafe.scalalogging.LazyLogging

import scala.util.Try

case class AnimationSpecifications(specification: List[String] = List.empty) extends LazyLogging:
  val parts: List[AnimationSpecification] = specification.map(AnimationSpecification(_))
  val numFrames: Int = parts.map(_.frames.getOrElse(0)).sum

  def valid(spongeType: String): Boolean =
    parts.forall(_.valid(spongeType)) && parts.map(_.seconds).map(_.isDefined).toSet.size < 2

  def rotationProjectionParameters(frame: Int): RotationProjectionParameters =
    def previousPlusCurrentRotation(specs: List[AnimationSpecification], frame: Int): RotationProjectionParameters =
      accumulateAllButLastRotationProjections(specs) + specs.last.rotationProjectionParameters(frame)

    partAndFrame(frame).map(previousPlusCurrentRotation).getOrElse(
      throw IllegalArgumentException("AnimationSpecification.frames not defined")
    )

  def isRotationAxisSet(x: Float, y: Float, z: Float, xw: Float, yw: Float, zw: Float): Boolean =
    parts.exists(_.isRotationAxisSet(x, y, z, xw, yw, zw))

  def accumulateAllButLastRotationProjections(specs: List[AnimationSpecification]): RotationProjectionParameters =
    specs.init.foldLeft(RotationProjectionParameters()) { (acc, spec) =>
      acc + spec.rotationProjectionParameters(spec.frames.getOrElse(0))
    }

  def partAndFrame(
                    totalFrame: Int,
                    partsParts: List[AnimationSpecification] = parts,
                    accumulator: List[AnimationSpecification] = List.empty
                  ): Try[(List[AnimationSpecification], Int)] =
    if partsParts.isEmpty then
      throw IllegalArgumentException("AnimationSpecification.parts not defined")
    else
      val current = partsParts.head
      if current.frames.isEmpty then
        throw IllegalArgumentException(s"Animation specification $current has no frames")
      else
        if current.frames.getOrElse(0) > totalFrame then
          Try((accumulator :+ current, totalFrame))
        else partAndFrame(totalFrame - current.frames.getOrElse(0), partsParts.tail, accumulator :+ current)
