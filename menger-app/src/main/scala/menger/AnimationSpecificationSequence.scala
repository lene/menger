package menger

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging
import menger.common.AnimationException

case class AnimationSpecificationSequence(specification: List[String] = List.empty) extends LazyLogging:
  val parts: List[AnimationSpecification] = specification.map(AnimationSpecification(_))
  val numFrames: Int = parts.map(_.frames.getOrElse(0)).sum
  require(parts.forall(_.isTimeSpecValid), "AnimationSpecification.frames not defined")

  def valid(spongeType: String): Boolean = parts.forall(_.valid(spongeType))
  def valid(spongeTypes: Set[String]): Boolean = parts.forall(_.valid(spongeTypes))
  def isTimeSpecValid: Boolean = parts.forall(_.isTimeSpecValid)

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

  def applyToSpec(spec: ObjectSpec, frame: Int): ObjectSpec =
    import AnimationSpecification.*
    val withLevel = level(frame).map(l => spec.copy(level = Some(l))).getOrElse(spec)
    val animatedParams = parts.flatMap(_.animationParameters.keys).toSet
    val has4D = animatedParams.exists(FOUR_D_VALID_PARAMETERS.contains)
    val has3D = animatedParams.exists(Set(RotX, RotY, RotZ).contains)
    if !has4D && !has3D then withLevel
    else
      val r = rotationProjectionParameters(frame).getOrElse(RotationProjectionParameters())
      val base = spec.projection4D.getOrElse(Projection4DSpec.default)
      val with4D = if has4D then
        withLevel.copy(projection4D = Some(Projection4DSpec(
          rotXW   = if animatedParams.contains(RotXW) then r.rotXW else base.rotXW,
          rotYW   = if animatedParams.contains(RotYW) then r.rotYW else base.rotYW,
          rotZW   = if animatedParams.contains(RotZW) then r.rotZW else base.rotZW,
          eyeW    = if animatedParams.contains(ProjectionEyeW) then r.eyeW else base.eyeW,
          screenW = if animatedParams.contains(ProjectionScreenW) then r.screenW else base.screenW
        )))
      else withLevel
      if has3D then
        val toRad = math.Pi.toFloat / 180f
        with4D.copy(rotation = ObjectRotation(
          x = if animatedParams.contains(RotX) then r.rotX * toRad else with4D.rotation.x,
          y = if animatedParams.contains(RotY) then r.rotY * toRad else with4D.rotation.y,
          z = if animatedParams.contains(RotZ) then r.rotZ * toRad else with4D.rotation.z
        ))
      else with4D

  def isRotationAxisSet(x: Float, y: Float, z: Float, xw: Float, yw: Float, zw: Float): Boolean =
    parts.exists(_.isRotationAxisSet(x, y, z, xw, yw, zw))

  def hasRotationAxisConflict(x: Float, y: Float, z: Float, xw: Float, yw: Float, zw: Float): Boolean =
    isRotationAxisSet(x, y, z, xw, yw, zw)

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
        scala.util.Failure(AnimationException("Animation specification parts not defined"))
      case current :: rest =>
        current.frames match
          case None =>
            val specIndex = parts.indexOf(current)
            scala.util.Failure(AnimationException(
              s"Animation specification has no frames: $current",
              Some(specIndex)
            ))
          case Some(frameCount) if frameCount > totalFrame =>
            Try((accumulator :+ current, totalFrame))
          case Some(frameCount) =>
            partAndFrame(totalFrame - frameCount, rest, accumulator :+ current)
