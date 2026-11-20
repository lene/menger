package examples.dsl.ladder

import examples.dsl.ladder.Choreography.Segment
import menger.dsl._

/** The plinth the object rests above and the camera that frames both, as `framing` says. The
  * camera stands with the sun behind it, so it needs the sun's direction; the object's centre
  * stays at one height, so that its lowest point over the whole cycle, `restDepth` below the
  * centre, just clears the plinth. */
final case class Stage(sunDirection: Vec3, restDepth: Float, framing: Framing = Framing.default):
  import Stage._

  val plinth: Cylinder =
    Cylinder(p0 = Some(Vec3(0f, plinthTopY - plinthThickness, 0f)),
      p1 = Some(Vec3(0f, plinthTopY, 0f)), radius = Some(plinthRadius),
      material = Some(Material.matte(plinthColor)))

  private val cameraDirection =
    val yaw = math.toRadians(cameraYawDegrees)
    val lift = math.toRadians(cameraElevationDegrees)
    val (x, z) = (sunDirection.x, sunDirection.z)
    val away = Vec3((x * math.cos(yaw) + z * math.sin(yaw)).toFloat, 0f,
      (-x * math.sin(yaw) + z * math.cos(yaw)).toFloat).normalize
    Vec3(away.x * math.cos(lift).toFloat, math.sin(lift).toFloat, away.z * math.cos(lift).toFloat)
  private val viewUp = (Vec3.UnitY - cameraDirection * cameraDirection.y).normalize
  private val plinthFrontBottom = Vec3(cameraDirection.x, 0f, cameraDirection.z).normalize *
    plinthRadius + Vec3(0f, plinthTopY - plinthThickness, 0f)

  /** Where the object's centre stays, at every t: its lowest point, in the lowest frame of the
    * cycle, is the clearance above the plinth. */
  val centre: Vec3 = Vec3(0f, plinthTopY + plinthClearance + restDepth, 0f)

  // The camera distance from `aim` at which `point` fits the frame: it projects at across /
  // (along + distance), `across` and `along` its offsets from the aim, at most the half tangent.
  private def fitDistance(point: Vec3, aim: Vec3, across: Vec3, halfTan: Float): Float =
    val offset = point - aim
    math.abs(offset.dot(across)) / (halfTan * framing.margin) + offset.dot(cameraDirection)

  // The height above the centre the camera must keep in the frame: what `fit` says of the object.
  private def reachAbove(dimension: Int, last: Float): Float = framing.fit match
    case Fit.Bound  => Choreography.boundingRadius(dimension, last)
    case Fit.Extent => Choreography.halfExtent(dimension, last)

  /** Aimed between the object and the plinth; as far as the object, plinth edge and width need. */
  def cameraAt(segment: Segment, u: Float): Camera =
    val last = Choreography.extrusion(segment, u)
    val aim = Vec3(0f, (centre.y + plinthTopY) / 2f, 0f)
    val objectTop = centre + Vec3(0f, reachAbove(segment.dimension, last), 0f)
    val dolly = Choreography.halfExtent(segment.dimension, last) / Choreography.halfExtent(3, 1f)
    val tanHalfFovX = framing.tanHalfFovX
    val tanHalfFovY = tanHalfFovX / frameAspect
    val distance = List(cameraDistance * dolly, plinthRadius / (tanHalfFovX * framing.margin),
      fitDistance(objectTop, aim, viewUp, tanHalfFovY),
      fitDistance(plinthFrontBottom, aim, viewUp, tanHalfFovY)).max
    Camera(position = aim + cameraDirection * distance, lookAt = aim)

object Stage:
  private[ladder] val plinthTopY = -0.45f
  private val plinthThickness = 0.2f
  private val plinthRadius = 1.8f
  // The warm grey the plinth had (0.85, 0.82, 0.76), scaled to a largest channel of 0.28, for
  // physical lighting (albedo * E / pi). Decided by user 2026-11-14: "3. 0.28/1.0".
  private val plinthColor = Color(0.28f, 0.27012f, 0.25035f)
  // 0.01 above the plinth's top cap: a bottom face on the cap itself speckles the refraction.
  private[ladder] val plinthClearance = 0.01f
  // How far below its centre the object reaches, edge tubes included, in the lowest frame of the
  // cycle of each top dimension: the minimum over every frame at 30 fps and both ends of every
  // segment, for `Look.default`'s breath depths (the Deep ones, menger#167) and edge radius,
  // rounded up to 1e-5. RestHeightSuite recomputes it. Recomputed over the derived 202.74 s cycle
  // (menger#160, the old pace): unchanged, as the deepest frames are segment ends.
  // The lowest point of a frame is its n-cube's lowest corner, and the cycle's lowest frame is a
  // solidifying cube mid-turn, so the object touches the plinth once per cycle and floats above
  // it otherwise.
  private val restDepths = Map(4 -> 0.95848f, 5 -> 1.11713f, 6 -> 1.34740f, 7 -> 1.66853f,
    8 -> 2.22143f)

  /** The depth below the centre to rest on for `top`; a thicker tube than the default look's
    * reaches `extraTube` further down. */
  private[ladder] def restDepth(top: Int, extraTube: Float = 0f): Float =
    restDepths(top) + math.max(0f, extraTube)

  // The sun behind the camera: 160 degrees round from the sun's rays puts the view 20 degrees off
  // them, 8 degrees down. The object is the star, not the sun, so it is lit face-on and its
  // shadows fall behind it, onto the plinth. Aimed between the object and the plinth. The field
  // of view of the `Framing` (45 degrees by default) is HORIZONTAL: fit the video's 16:9 frame,
  // 9/16 as tall as wide.
  private val cameraDistance = 4.5f
  private val cameraYawDegrees = 160f
  private val cameraElevationDegrees = 8f
  private val frameAspect = 16f / 9f
