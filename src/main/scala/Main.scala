package menger

import com.badlogic.gdx.backends.lwjgl3.{
  Lwjgl3Application, Lwjgl3ApplicationConfiguration
}


object Main extends App:
  private final val COLOR_BITS = 8
  private final val DEPTH_BITS = 16
  private final val STENCIL_BITS = 0
  private final val NUM_ANTIALIAS_SAMPLES = 4

  val config = new Lwjgl3ApplicationConfiguration
  config.disableAudio(true)
  config.setTitle("Engine Test")
  config.setWindowedMode(800, 600)
  config.setBackBufferConfig(
    COLOR_BITS, COLOR_BITS, COLOR_BITS, COLOR_BITS, DEPTH_BITS, STENCIL_BITS,
    NUM_ANTIALIAS_SAMPLES
  )
  new Lwjgl3Application(new EngineTest, config)
