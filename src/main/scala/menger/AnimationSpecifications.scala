package menger

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging

case class AnimationSpecifications(specification: List[String] = List.empty) extends LazyLogging:
  val parts: List[AnimationSpecification] = specification.map(AnimationSpecification(_))
  val numFrames: Int = parts.map(_.frames.getOrElse(0)).sum
  require(parts.forall(_.timeSpecValid), "AnimationSpecification.frames not defined")

  def valid(spongeType: String): Boolean = parts.forall(_.valid(spongeType))
  def timeSpecValid: Boolean = parts.forall(_.timeSpecValid)

  def level(frame: Int): Option[Float] =
    partAndFrame(frame).toOption.flatMap { case (specs, frameOffset) =>
      specs.last.level(frameOffset)
    }

  def rotationProjectionParameters(frame: Int): Try[RotationProjectionParameters] =
    def previousPlusCurrentRotation(specs: List[AnimationSpecification], frame: Int): RotationProjectionParameters =
      accumulateAllButLastRotationProjections(specs) + specs.last.rotationProjectionParameters(frame)

    partAndFrame(frame).map { case (specs, frameOffset) =>
      previousPlusCurrentRotation(specs, frameOffset)
    }

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
    partsParts match
      case Nil =>
        scala.util.Failure(IllegalArgumentException("AnimationSpecification.parts not defined"))
      case current :: rest =>
        current.frames match
          case None =>
            scala.util.Failure(IllegalArgumentException(s"Animation specification $current has no frames"))
          case Some(frameCount) if frameCount > totalFrame =>
            Try((accumulator :+ current, totalFrame))
          case Some(frameCount) =>
            partAndFrame(totalFrame - frameCount, rest, accumulator :+ current)
