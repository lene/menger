package menger.input

import scala.math._

import com.badlogic.gdx.math.Vector3
import menger.common.Const

case class OrbitConfig(
  orbitSensitivity: Float = 0.3f,
  panSensitivity: Float = 0.005f,
  zoomSensitivity: Float = 0.1f,
  minDistance: Float = 0.5f,
  maxDistance: Float = 20.0f,
  minElevation: Float = -89.0f,
  maxElevation: Float = 89.0f
)

trait SphericalOrbit:
  protected def orbitConfig: OrbitConfig

  // Spherical coordinates state (subclasses must provide storage)
  protected def azimuth: Float
  protected def azimuth_=(value: Float): Unit
  protected def elevation: Float
  protected def elevation_=(value: Float): Unit
  protected def distance: Float
  protected def distance_=(value: Float): Unit

  // Convert eye/lookAt to initial spherical coordinates
  protected def initSpherical(eye: Vector3, lookAt: Vector3): (Float, Float, Float) =
    val dir = eye.cpy().sub(lookAt)
    val dist = dir.len()
    val azimuthRad = atan2(dir.x.toDouble, dir.z.toDouble).toFloat
    val horizontalDist = sqrt(dir.x * dir.x + dir.z * dir.z).toFloat
    val elevationRad = atan2(dir.y.toDouble, horizontalDist.toDouble).toFloat
    (Const.radiansToDegrees(azimuthRad), Const.radiansToDegrees(elevationRad), dist)

  // Update spherical coords from mouse delta (orbit)
  protected def updateOrbit(deltaX: Int, deltaY: Int): Unit =
    azimuth = azimuth + deltaX * orbitConfig.orbitSensitivity
    val newElev = elevation - deltaY * orbitConfig.orbitSensitivity
    elevation = newElev.max(orbitConfig.minElevation).min(orbitConfig.maxElevation)

  // Update distance from scroll (zoom)
  protected def updateZoom(scrollAmount: Float): Unit =
    val zoomFactor = 1.0f + (scrollAmount * orbitConfig.zoomSensitivity)
    val newDist = distance * zoomFactor
    distance = newDist.max(orbitConfig.minDistance).min(orbitConfig.maxDistance)

  // Convert spherical back to Cartesian eye position
  protected def sphericalToCartesian(lookAt: Vector3): Vector3 =
    val azimuthRad = Const.degreesToRadians(azimuth)
    val elevationRad = Const.degreesToRadians(elevation)
    val cosElev = cos(elevationRad.toDouble).toFloat
    val x = lookAt.x + distance * sin(azimuthRad.toDouble).toFloat * cosElev
    val y = lookAt.y + distance * sin(elevationRad.toDouble).toFloat
    val z = lookAt.z + distance * cos(azimuthRad.toDouble).toFloat * cosElev
    Vector3(x, y, z)

  // Compute pan offset from mouse delta
  protected def computePanOffset(deltaX: Int, deltaY: Int, forward: Vector3, up: Vector3): Vector3 =
    val right = forward.cpy().crs(up).nor()
    val camUp = right.cpy().crs(forward).nor()
    val panScale = distance * orbitConfig.panSensitivity
    val deltaRight = right.scl(deltaX * panScale)
    val deltaUp = camUp.scl(-deltaY * panScale)
    deltaRight.add(deltaUp)
