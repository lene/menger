package menger.engines

import menger.ProfilingConfig
import menger.optix.CameraState
import menger.optix.SceneConfigurator

// Stub: constructor signature and implementation are provided by Task 17.6
class PreviewEngine()(using ProfilingConfig)
    extends BaseEngine(0) with WithPreview:

  override protected def textureDir: String = "."

  override protected val sceneConfigurator: SceneConfigurator =
    sys.error("PreviewEngine not yet implemented (Task 17.6)")

  override protected val cameraState: CameraState =
    sys.error("PreviewEngine not yet implemented (Task 17.6)")

  override def create(): Unit = sys.error("PreviewEngine not yet implemented (Task 17.6)")
  override def render(): Unit = {}
