package menger.optix

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import ImageMatchers.*

class ShadowTest extends AnyFlatSpec with Matchers with RendererFixture:

  "Shadow rays" should "create darker shadows when enabled" in:
    // Setup basic scene
    renderer.setSphere(0.0f, 0.0f, 0.0f, 0.5f)
    renderer.setSphereColor(0.75f, 0.75f, 0.75f, 1.0f)  // Opaque sphere
    renderer.setIOR(1.5f)
    renderer.setScale(1.0f)
    renderer.setPlane(1, true, -0.6f)  // Y-axis plane below sphere

    val eye = Array(0.0f, 0.0f, 5.0f)
    val lookAt = Array(0.0f, -0.3f, 0.0f)
    val up = Array(0.0f, 1.0f, 0.0f)
    renderer.setCamera(eye, lookAt, up, 45.0f)

    val lightDir = Array(0.5f, 0.5f, -0.5f)
    renderer.setLight(lightDir, 1.0f)

    // Render without shadows
    renderer.setShadows(false)
    val imageWithoutShadows = renderer.render(800, 600).get

    // Render with shadows
    renderer.setShadows(true)
    val imageWithShadows = renderer.render(800, 600).get

    // Images should be different (shadows should darken the plane)
    java.util.Arrays.equals(imageWithoutShadows, imageWithShadows) shouldBe false

  it should "respect material transparency" in:
    // Setup scene
    renderer.setSphere(0.0f, 0.0f, 0.0f, 0.5f)
    renderer.setIOR(1.5f)
    renderer.setScale(1.0f)
    renderer.setPlane(1, true, -0.6f)

    val eye = Array(0.0f, 0.0f, 5.0f)
    val lookAt = Array(0.0f, -0.3f, 0.0f)
    val up = Array(0.0f, 1.0f, 0.0f)
    renderer.setCamera(eye, lookAt, up, 45.0f)

    val lightDir = Array(0.5f, 0.5f, -0.5f)
    renderer.setLight(lightDir, 1.0f)
    renderer.setShadows(true)

    // Render with fully transparent sphere (alpha=0.0)
    renderer.setSphereColor(0.75f, 0.75f, 0.75f, 0.0f)
    val transparentImage = renderer.render(800, 600).get

    // Render with opaque sphere (alpha=1.0)
    renderer.setSphereColor(0.75f, 0.75f, 0.75f, 1.0f)
    val opaqueImage = renderer.render(800, 600).get

    // Images should be different (opaque sphere casts darker shadow)
    java.util.Arrays.equals(transparentImage, opaqueImage) shouldBe false
