package menger.optix

import org.scalatest.matchers.{BeMatcher, MatchResult, Matcher}
import ThresholdConstants.*


object ImageMatchers:

  
  private def imageMatchResult(
    matches: Boolean,
    rawFailureMessage: String,
    rawNegatedFailureMessage: String,
    width: Int,
    height: Int
  ): MatchResult =
    MatchResult(
      matches,
      s"$rawFailureMessage (image: ${width}x${height})",
      s"$rawNegatedFailureMessage (image: ${width}x${height})"
    )

  // ========== Image Size Matchers ==========

  
  def haveValidRGBASize(width: Int, height: Int): Matcher[Array[Byte]] =
    Matcher { imageData =>
      val expectedSize = ImageValidation.imageByteSize(width, height)
      val actualSize = imageData.length
      val matches = actualSize == expectedSize

      MatchResult(
        matches,
        s"Image size was $actualSize bytes but expected $expectedSize bytes for ${width}x${height} RGBA",
        s"Image size was $expectedSize bytes for ${width}x${height} RGBA as expected"
      )
    }

  def haveValidRGBASize(size: ImageSize): Matcher[Array[Byte]] =
    haveValidRGBASize(size.width, size.height)

  // ========== Sphere Center Matchers ==========

  
  def haveSphereCenterAt(
    expectedX: Int,
    expectedY: Int,
    width: Int,
    height: Int,
    tolerance: Int = SPHERE_CENTER_TOLERANCE
  ): Matcher[Array[Byte]] =
    Matcher { imageData =>
      val (actualX, actualY) = ImageValidation.detectSphereCenter(imageData, width, height)
      val xMatch = math.abs(actualX - expectedX) <= tolerance
      val yMatch = math.abs(actualY - expectedY) <= tolerance
      val matches = xMatch && yMatch

      imageMatchResult(
        matches,
        s"Sphere center at ($actualX, $actualY) was not within $tolerance pixels of expected ($expectedX, $expectedY)",
        s"Sphere center at ($actualX, $actualY) was within $tolerance pixels of ($expectedX, $expectedY) but should not have been",
        width,
        height
      )
    }

  
  def haveSphereCenteredInImage(
    width: Int,
    height: Int,
    tolerance: Int = SPHERE_CENTER_TOLERANCE
  ): Matcher[Array[Byte]] =
    haveSphereCenterAt(width / 2, height / 2, width, height, tolerance)

  // ========== Sphere Area Matchers ==========

  
  def haveSphereAreaBetween(
    minArea: Int,
    maxArea: Int,
    width: Int,
    height: Int
  ): Matcher[Array[Byte]] =
    Matcher { imageData =>
      val area = ImageValidation.spherePixelArea(imageData, width, height)
      val matches = area >= minArea && area <= maxArea

      imageMatchResult(
        matches,
        s"Sphere area $area was not between $minArea and $maxArea pixels",
        s"Sphere area $area was between $minArea and $maxArea pixels but should not have been",
        width,
        height
      )
    }

  
  def haveSmallSphereArea(width: Int, height: Int): Matcher[Array[Byte]] =
    Matcher { imageData =>
      val area = ImageValidation.spherePixelArea(imageData, width, height)
      val matches = area < SMALL_SPHERE_MAX_AREA

      imageMatchResult(
        matches,
        s"Sphere area $area was not less than $SMALL_SPHERE_MAX_AREA pixels (small sphere threshold)",
        s"Sphere area $area was less than $SMALL_SPHERE_MAX_AREA pixels but should not have been",
        width,
        height
      )
    }

  def haveSmallSphereArea(size: ImageSize): Matcher[Array[Byte]] =
    haveSmallSphereArea(size.width, size.height)


  def haveMediumSphereArea(width: Int, height: Int): Matcher[Array[Byte]] =
    haveSphereAreaBetween(MEDIUM_SPHERE_MIN_AREA, MEDIUM_SPHERE_MAX_AREA, width, height)

  def haveMediumSphereArea(size: ImageSize): Matcher[Array[Byte]] =
    haveMediumSphereArea(size.width, size.height)


  def haveLargeSphereArea(width: Int, height: Int): Matcher[Array[Byte]] =
    Matcher { imageData =>
      val area = ImageValidation.spherePixelArea(imageData, width, height)
      val matches = area > LARGE_SPHERE_MIN_AREA

      imageMatchResult(
        matches,
        s"Sphere area $area was not greater than $LARGE_SPHERE_MIN_AREA pixels (large sphere threshold)",
        s"Sphere area $area was greater than $LARGE_SPHERE_MIN_AREA pixels but should not have been",
        width,
        height
      )
    }

  def haveLargeSphereArea(size: ImageSize): Matcher[Array[Byte]] =
    haveLargeSphereArea(size.width, size.height)

  // ========== Color Matchers ==========

  
  def haveDominantChannel(
    expectedChannel: String,
    width: Int,
    height: Int,
    tolerance: Double = GRAYSCALE_CHANNEL_TOLERANCE
  ): Matcher[Array[Byte]] =
    Matcher { imageData =>
      val actualChannel = ImageValidation.dominantColorChannel(
        imageData,
        width,
        height,
        tolerance
      )
      val matches = actualChannel == expectedChannel

      imageMatchResult(
        matches,
        s"Dominant color channel was '$actualChannel' but expected '$expectedChannel'",
        s"Dominant color channel was '$expectedChannel' but should not have been",
        width,
        height
      )
    }

  
  def beRedDominant(width: Int, height: Int, tolerance: Double = GRAYSCALE_CHANNEL_TOLERANCE): Matcher[Array[Byte]] =
    haveDominantChannel("r", width, height, tolerance)

  def beRedDominant(size: ImageSize): Matcher[Array[Byte]] =
    beRedDominant(size.width, size.height)

  def beRedDominant(size: ImageSize, tolerance: Double): Matcher[Array[Byte]] =
    beRedDominant(size.width, size.height, tolerance)


  def beGreenDominant(width: Int, height: Int, tolerance: Double = GRAYSCALE_CHANNEL_TOLERANCE): Matcher[Array[Byte]] =
    haveDominantChannel("g", width, height, tolerance)

  def beGreenDominant(size: ImageSize): Matcher[Array[Byte]] =
    beGreenDominant(size.width, size.height)

  def beGreenDominant(size: ImageSize, tolerance: Double): Matcher[Array[Byte]] =
    beGreenDominant(size.width, size.height, tolerance)


  def beBlueDominant(width: Int, height: Int, tolerance: Double = GRAYSCALE_CHANNEL_TOLERANCE): Matcher[Array[Byte]] =
    haveDominantChannel("b", width, height, tolerance)

  def beBlueDominant(size: ImageSize): Matcher[Array[Byte]] =
    beBlueDominant(size.width, size.height)

  def beBlueDominant(size: ImageSize, tolerance: Double): Matcher[Array[Byte]] =
    beBlueDominant(size.width, size.height, tolerance)


  def beGrayscale(width: Int, height: Int): Matcher[Array[Byte]] =
    haveDominantChannel("gray", width, height)

  def beGrayscale(size: ImageSize): Matcher[Array[Byte]] =
    beGrayscale(size.width, size.height)


  def haveColorRatio(
    expectedR: Double,
    expectedG: Double,
    expectedB: Double,
    width: Int,
    height: Int,
    tolerance: Double = 0.1
  ): Matcher[Array[Byte]] =
    Matcher { imageData =>
      val (r, g, b) = ImageValidation.colorChannelRatio(imageData, width, height)
      val rMatch = math.abs(r - expectedR) <= tolerance
      val gMatch = math.abs(g - expectedG) <= tolerance
      val bMatch = math.abs(b - expectedB) <= tolerance
      val matches = rMatch && gMatch && bMatch

      imageMatchResult(
        matches,
        f"Color ratio (R=$r%.3f, G=$g%.3f, B=$b%.3f) was not within $tolerance of expected (R=$expectedR%.3f, G=$expectedG%.3f, B=$expectedB%.3f)",
        f"Color ratio (R=$r%.3f, G=$g%.3f, B=$b%.3f) was within $tolerance of (R=$expectedR%.3f, G=$expectedG%.3f, B=$expectedB%.3f) but should not have been",
        width,
        height
      )
    }

  // ========== Brightness Matchers ==========

  
  def haveBrightnessStdDevGreaterThan(
    threshold: Double,
    width: Int,
    height: Int
  ): Matcher[Array[Byte]] =
    Matcher { imageData =>
      val stdDev = ImageValidation.brightnessStdDev(imageData, width, height)
      val matches = stdDev > threshold

      imageMatchResult(
        matches,
        f"Brightness standard deviation $stdDev%.2f was not greater than $threshold%.2f",
        f"Brightness standard deviation $stdDev%.2f was greater than $threshold%.2f but should not have been",
        width,
        height
      )
    }

  
  def showGlassRefraction(width: Int, height: Int): Matcher[Array[Byte]] =
    haveBrightnessStdDevGreaterThan(MIN_GLASS_REFRACTION_STDDEV, width, height)

  def showGlassRefraction(size: ImageSize): Matcher[Array[Byte]] =
    showGlassRefraction(size.width, size.height)


  def showWaterRefraction(width: Int, height: Int): Matcher[Array[Byte]] =
    haveBrightnessStdDevGreaterThan(MIN_WATER_REFRACTION_STDDEV, width, height)

  def showWaterRefraction(size: ImageSize): Matcher[Array[Byte]] =
    showWaterRefraction(size.width, size.height)


  def showDiamondRefraction(width: Int, height: Int): Matcher[Array[Byte]] =
    haveBrightnessStdDevGreaterThan(MIN_DIAMOND_REFRACTION_STDDEV, width, height)

  def showDiamondRefraction(size: ImageSize): Matcher[Array[Byte]] =
    showDiamondRefraction(size.width, size.height)

  // ========== Background/Plane Visibility Matchers ==========


  def showBackground(width: Int, height: Int): Matcher[Array[Byte]] =
    Matcher { imageData =>
      val visible = ImageValidation.backgroundVisibility(imageData, width, height)

      imageMatchResult(
        visible,
        "Background checkered pattern was not detected (sphere might be opaque)",
        "Background checkered pattern was detected but should not have been",
        width,
        height
      )
    }

  def showBackground(size: ImageSize): Matcher[Array[Byte]] =
    showBackground(size.width, size.height)


  def showPlaneInRegion(
    region: String,
    width: Int,
    height: Int
  ): Matcher[Array[Byte]] =
    Matcher { imageData =>
      val visible = ImageValidation.planeVisibility(imageData, width, height, region)

      imageMatchResult(
        visible,
        s"Checkered plane pattern was not detected in $region region",
        s"Checkered plane pattern was detected in $region region but should not have been",
        width,
        height
      )
    }
