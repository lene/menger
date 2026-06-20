package menger.engines

import menger.CurveData
import menger.ObjectSpec
import menger.common.ProfilingConfig
import menger.engines.scene.CurveSceneBuilder
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CurveRoutingSuite extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  private val curveSpec = ObjectSpec(
    objectType = "curve",
    curveData = Some(CurveData(
      points = Vector(0f, 0f, 0f, 1f, 0f, 0f, 1f, 1f, 0f, 0f, 1f, 0f),
      widths = Vector(0.05f, 0.05f, 0.05f, 0.05f)
    ))
  )

  "GeometryRegistry" should "return CurveSceneBuilder for curve specs" in:
    val builder = GeometryRegistry.builderFor(List(curveSpec))
    builder shouldBe defined
    builder.get shouldBe a[CurveSceneBuilder]

  "RenderModeSelector" should "classify curve-only scene as Curves" in:
    RenderModeSelector.classify(List(curveSpec)) shouldBe SceneType.Curves(List(curveSpec))
