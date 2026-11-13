package examples.dsl.ladder

import examples.dsl.ladder.Choreography.Segment
import menger.dsl._

/** The plinth the object rests above and the camera that frames both. The camera stands on the far
  * side from the sun, so it needs the sun's direction; the rest height needs the edge tubes' radius. */
final case class Stage(sunDirection: Vec3, edgeRadius: Float):
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

  /** Where the object rests: its bounding radius, edge tubes and clearance above the plinth. */
  def centreAt(segment: Segment, u: Float): Vec3 =
    val radius = Choreography.boundingRadius(segment.dimension, Choreography.extrusion(segment, u))
    Vec3(0f, plinthTopY + plinthClearance + edgeRadius + radius, 0f)

  // The camera distance from `aim` at which `point` fits the frame: it projects at across /
  // (along + distance), `across` and `along` its offsets from the aim, at most the half tangent.
  private def fitDistance(point: Vec3, aim: Vec3, across: Vec3, halfTan: Float): Float =
    val offset = point - aim
    math.abs(offset.dot(across)) / (halfTan * framingMargin) + offset.dot(cameraDirection)

  /** Aimed between the object and the plinth; as far as the object, plinth edge and width need. */
  def cameraAt(segment: Segment, u: Float): Camera =
    val (last, centre) = (Choreography.extrusion(segment, u), centreAt(segment, u))
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
  private val plinthClearance = 0.01f
  // Low, on the far side from the sun (along its rays), 15 degrees to one side, aimed between the
  // object and the plinth. The fixed 45 degree field of view is HORIZONTAL (tan 22.5 = 0.4142): fit
  // the video's 16:9 frame, 9/16 as tall as wide.
  private val cameraDistance = 4.5f
  private val cameraYawDegrees = 15f
  private val cameraElevationDegrees = 4f
  private val tanHalfFovX = 0.4142f
  private val frameAspect = 16f / 9f
  private val framingMargin = 0.95f
