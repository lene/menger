package examples.dsl.ladder

import examples.dsl.ladder.Choreography.Segment
import menger.dsl._

/** The plinth the object rests above and the camera that frames both. The camera stands with the
  * sun behind it, so it needs the sun's direction; the object's centre stays at one height, so that its
  * lowest point over the whole cycle, `restDepth` below the centre, just clears the plinth. */
final case class Stage(sunDirection: Vec3, restDepth: Float):
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
    math.abs(offset.dot(across)) / (halfTan * framingMargin) + offset.dot(cameraDirection)

  /** Aimed between the object and the plinth; as far as the object, plinth edge and width need. */
  def cameraAt(segment: Segment, u: Float): Camera =
    val last = Choreography.extrusion(segment, u)
    val aim = Vec3(0f, (centre.y + plinthTopY) / 2f, 0f)
    val objectTop = centre + Vec3(0f, Choreography.boundingRadius(segment.dimension, last), 0f)
    val dolly = Choreography.halfExtent(segment.dimension, last) / Choreography.halfExtent(3, 1f)
    val tanHalfFovY = tanHalfFovX / frameAspect
    val distance = List(cameraDistance * dolly, plinthRadius / (tanHalfFovX * framingMargin),
      fitDistance(objectTop, aim, viewUp, tanHalfFovY),
      fitDistance(plinthFrontBottom, aim, viewUp, tanHalfFovY)).max
    Camera(position = aim + cameraDirection * distance, lookAt = aim)

object Stage:
  private val plinthTopY = -0.45f
  private val plinthThickness = 0.2f
  private val plinthRadius = 1.8f
  private val plinthColor = Color(0.85f, 0.82f, 0.76f)
  // 0.01 above the plinth's top cap: a bottom face on the cap itself speckles the refraction.
  private[ladder] val plinthClearance = 0.01f
  // How far below its centre the object reaches, edge tubes included, in the lowest frame of the
  // cycle of each top dimension: the minimum over every frame at 30 fps and both ends of every
  // segment, for `Look.default`'s breath depths and edge radius. RestHeightSuite recomputes it.
  // The lowest point of a frame is its n-cube's lowest corner, and the cycle's lowest frame is a
  // solidifying cube mid-turn, so the object touches the plinth once per cycle and floats above
  // it otherwise.
  private val restDepths = Map(4 -> 0.96035f, 5 -> 1.12766f, 6 -> 1.33819f, 7 -> 1.66853f,
    8 -> 2.22141f)

  /** The depth below the centre to rest on for `top`; a thicker tube than the default look's
    * reaches `extraTube` further down. */
  private[ladder] def restDepth(top: Int, extraTube: Float = 0f): Float =
    restDepths(top) + math.max(0f, extraTube)

  // The sun behind the camera: 160 degrees round from the sun's rays puts the view 20 degrees off
  // them, 8 degrees down. The object is the star, not the sun, so it is lit face-on and its
  // shadows fall behind it, onto the plinth. Aimed between the object and the plinth. The fixed
  // 45 degree field of view is HORIZONTAL (tan 22.5 = 0.4142): fit the video's 16:9 frame, 9/16
  // as tall as wide.
  private val cameraDistance = 4.5f
  private val cameraYawDegrees = 160f
  private val cameraElevationDegrees = 8f
  private val tanHalfFovX = 0.4142f
  private val frameAspect = 16f / 9f
  private val framingMargin = 0.95f
