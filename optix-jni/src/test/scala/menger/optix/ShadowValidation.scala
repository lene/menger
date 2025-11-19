package menger.optix

import ImageValidation.*
import scala.math.{abs, sqrt}


object ShadowValidation:

  
  case class Region(x0: Int, y0: Int, x1: Int, y1: Int):
    def width: Int = x1 - x0
    def height: Int = y1 - y0
    def area: Int = width * height
    def contains(x: Int, y: Int): Boolean =
      x >= x0 && x < x1 && y >= y0 && y < y1

  object Region:
    
    def centered(cx: Int, cy: Int, radius: Int): Region =
      Region(cx - radius, cy - radius, cx + radius, cy + radius)

    
    def bottomCenter(width: Int, height: Int, fraction: Double = 0.25): Region =
      val regionWidth = (width * fraction).toInt
      val regionHeight = (height * fraction).toInt
      val x0 = (width - regionWidth) / 2
      val y0 = height - regionHeight
      Region(x0, y0, x0 + regionWidth, y0 + regionHeight)

    
    def topCenter(width: Int, height: Int, fraction: Double = 0.25): Region =
      val regionWidth = (width * fraction).toInt
      val regionHeight = (height * fraction).toInt
      val x0 = (width - regionWidth) / 2
      Region(x0, 0, x0 + regionWidth, regionHeight)

    
    def leftSide(width: Int, height: Int, fraction: Double = 0.25): Region =
      val regionWidth = (width * fraction).toInt
      val y0 = height / 4
      val y1 = 3 * height / 4
      Region(0, y0, regionWidth, y1)

    
    def rightSide(width: Int, height: Int, fraction: Double = 0.25): Region =
      val regionWidth = (width * fraction).toInt
      val y0 = height / 4
      val y1 = 3 * height / 4
      Region(width - regionWidth, y0, width, y1)

  
  def regionBrightness(
    imageData: Array[Byte],
    width: Int,
    height: Int,
    region: Region
  ): Double =
    val samples = for
      y <- region.y0 until region.y1 if y >= 0 && y < height
      x <- region.x0 until region.x1 if x >= 0 && x < width
    yield getRGBAt(imageData, width, x, y).brightness

    if samples.isEmpty then 0.0
    else samples.sum.toDouble / samples.length

  
  def brightnessContrast(
    imageData: Array[Byte],
    width: Int,
    height: Int,
    region1: Region,
    region2: Region
  ): Double =
    val b1 = regionBrightness(imageData, width, height, region1)
    val b2 = regionBrightness(imageData, width, height, region2)
    if b2 == 0.0 then 0.0 else b1 / b2

  
  def detectDarkestRegion(
    imageData: Array[Byte],
    width: Int,
    height: Int,
    gridSize: Int = 5
  ): Region =
    val yStart = height / 2
    val regionWidth = width / gridSize
    val regionHeight = (height - yStart) / gridSize

    val regions = for
      gy <- 0 until gridSize
      gx <- 0 until gridSize
      x0 = gx * regionWidth
      y0 = yStart + gy * regionHeight
      x1 = x0 + regionWidth
      y1 = y0 + regionHeight
      region = Region(x0, y0, x1, y1)
    yield region

    regions.minBy(region => regionBrightness(imageData, width, height, region))

  
  def expectedShadowPosition(
    sphereX: Float,
    sphereY: Float,
    sphereZ: Float,
    lightDirX: Float,
    lightDirY: Float,
    lightDirZ: Float,
    planeY: Float
  ): (Float, Float) =
    // Avoid division by zero
    if abs(lightDirY) < 0.001f then
      (sphereX, sphereZ)
    else
      val t = (sphereY - planeY) / lightDirY
      val shadowX = sphereX - lightDirX * t
      val shadowZ = sphereZ - lightDirZ * t
      (shadowX, shadowZ)

  
  def shadowIntensity(litBrightness: Double, shadowedBrightness: Double): Double =
    if litBrightness == 0.0 then 0.0
    else 1.0 - (shadowedBrightness / litBrightness)

  
  def expectedShadowIntensity(alpha: Float): Double =
    // Maximum shadow intensity (opaque sphere) is about 0.74 based on shader constants
    val maxShadowIntensity = 0.74
    alpha * maxShadowIntensity
