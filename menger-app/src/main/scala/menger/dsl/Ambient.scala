package menger.dsl

import scala.util.Try

/** How the ambient fill is computed (menger#175): `Constant` is the same everywhere,
  * `Occlusion` darkens cavities and contact areas by the share of the hemisphere that is
  * covered. A scene that sets neither keeps the renderer's own default, `Occlusion()`.
  *
  * `radius` is how far from a surface an occluder still counts, in world units; the default,
  * `Float.PositiveInfinity`, counts every occluder. An infinite floor then covers half the
  * hemisphere of every vertical face, which is the noisiest case: a finite radius darkens
  * only what is near. */
sealed trait Ambient

object Ambient:

  case object Constant extends Ambient

  final case class Occlusion(
    samples: Int = Occlusion.DefaultSamples,
    radius: Float = Float.PositiveInfinity
  ) extends Ambient:
    FieldConstraints.check("Occlusion", "samples", samples)
    FieldConstraints.checkFloat("Occlusion", "radius", radius)

  object Occlusion:
    val DefaultSamples = 16

  private val ConstantName = "constant"
  private val OcclusionName = "occlusion"
  private val UnlimitedRadiusName = "inf"
  private val Separator = ':'
  private val Part = s"([^$Separator]*)"
  private val OcclusionPattern =
    s"$OcclusionName(?:$Separator$Part)?(?:$Separator$Part)?".r

  /** Reads `constant` or `occlusion[:samples[:radius]]`, the form of `--ambient`; a radius of
    * `inf` is unlimited. The error names the offending value and its range. */
  def parse(spec: String): Either[String, Ambient] =
    spec.trim.toLowerCase match
      case ConstantName => Right(Constant)
      case OcclusionPattern(samples, radius) =>
        occlusionOf(Option(samples).filter(_.nonEmpty), Option(radius).filter(_.nonEmpty), spec)
      case _ => Left(s"--ambient '$spec' is not recognized: expected '$ConstantName' or " +
        s"'$OcclusionName[:samples[:radius]]'")

  private def occlusionOf(
    samplesText: Option[String], radiusText: Option[String], spec: String
  ): Either[String, Ambient] =
    val defaults = Occlusion()
    for
      samples <- samplesText.fold(Right(defaults.samples))(parseSamples(_, spec))
      radius <- radiusText.fold(Right(defaults.radius))(parseRadius(_, spec))
      occlusion <- Try(Occlusion(samples, radius)).toEither.left.map(e => s"--ambient '$spec': " +
        Option(e.getMessage).getOrElse("out of range"))
    yield occlusion

  private def parseSamples(text: String, spec: String): Either[String, Int] =
    text.toIntOption.toRight(s"--ambient '$spec': samples '$text' is not a whole number " +
      s"(Occlusion.samples must be 1-${menger.dsl.ResourceLimits.ambientSamplesMax})")

  private def parseRadius(text: String, spec: String): Either[String, Float] =
    if text == UnlimitedRadiusName then Right(Float.PositiveInfinity)
    else text.toFloatOption.toRight(s"--ambient '$spec': radius '$text' is not a number " +
      s"(Occlusion.radius must be >= ${menger.dsl.ResourceLimits.ambientRadiusMin} or " +
      s"$UnlimitedRadiusName)")

  /** The inverse of `parse`, for messages and logs. */
  def describe(ambient: Ambient): String = ambient match
    case Constant => ConstantName
    case Occlusion(samples, radius) =>
      val radiusText =
        if radius.isInfinite then UnlimitedRadiusName else radius.toString
      s"$OcclusionName$Separator$samples$Separator$radiusText"
