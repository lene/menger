package io.github.lene.optix

import menger.common.Color
import menger.common.FogConfig
import menger.common.Light
import menger.common.Vector
import org.scalamock.scalatest.MockFactory
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneConfiguratorSuite extends AnyFlatSpec with Matchers with MockFactory:

  private val pos    = Vector[3](0f, 0f, 5f)
  private val lookAt = Vector[3](0f, 0f, 0f)
  private val up     = Vector[3](0f, 1f, 0f)

  private def configurator(lights: Array[Light] = Array.empty) =
    new SceneConfigurator(pos, lookAt, up, lights)

  "configureCamera" should "apply the configured camera" in:
    val renderer = mock[OptiXRenderer]
    (renderer.setCamera _).expects(pos, lookAt, up, *).once()
    configurator().configureCamera(renderer)

  "configureLights" should "forward an explicit light list via setLights" in:
    val renderer = mock[OptiXRenderer]
    val lights = Array[Light](Light.Directional(Vector[3](-1f, 1f, -1f), Color(1f, 1f, 1f), 1.0f))
    (renderer.setLights(_: Array[Light])).expects(lights).once()
    configurator(lights).configureLights(renderer)

  it should "fall back to a single default directional light when none are given" in:
    val renderer = mock[OptiXRenderer]
    (renderer.setLight(_: Vector[3], _: Float)).expects(*, *).once()
    configurator().configureLights(renderer)

  "setBackgroundColor" should "forward the RGB components" in:
    val renderer = mock[OptiXRenderer]
    (renderer.setBackgroundColor _).expects(0.1f, 0.2f, 0.3f).once()
    configurator().setBackgroundColor(renderer, Color(0.1f, 0.2f, 0.3f))

  "setFog" should "forward density and color components" in:
    val renderer = mock[OptiXRenderer]
    (renderer.setFog _).expects(0.5f, 0.1f, 0.2f, 0.3f).once()
    configurator().setFog(renderer, FogConfig(0.5f, Color(0.1f, 0.2f, 0.3f)))
