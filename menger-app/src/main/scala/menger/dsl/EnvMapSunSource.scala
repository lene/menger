package menger.dsl

import java.util.concurrent.atomic.AtomicReference

/** The sun of an environment map as a scene sees it: where it is, how strong it is, what colour.
  *
  * @param directionToSun unit vector from the scene to the sun, in the map's frame
  * @param halfAngleDegrees half-angle of the cone around `directionToSun` that is the sun
  * @param intensity intensity of a directional light that replaces the sun
  * @param color colour of that light, 1 in its strongest channel
  */
final case class EnvMapSunMeasurement(
  directionToSun: Vec3,
  halfAngleDegrees: Float,
  intensity: Float,
  color: Color
)

/** Where `Sun.fromEnvMap` gets its measurement from. The DSL cannot read files or depend on the
  * renderer (it is all a scene sees of the program), so whoever can reads them and installs the
  * lookup here: the renderer's `Main`, the scene validator. Without one, no map has a sun: a
  * scene still builds, with a stand-in sun.
  */
object EnvMapSunSource:

  private val lookup = new AtomicReference[String => Option[EnvMapSunMeasurement]](_ => None)

  /** From now on `measure` asks `lookup`: the sun of the map at a path relative to the texture
    * directory, or `None` when there is no such map.
    */
  private[menger] def install(newLookup: String => Option[EnvMapSunMeasurement]): Unit =
    lookup.set(newLookup)

  private[menger] def measure(path: String): Option[EnvMapSunMeasurement] = lookup.get.apply(path)
