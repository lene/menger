package menger.engines

trait SavesScreenshots:
  protected def saveImage(): Unit =
    currentSaveName.foreach(ScreenshotFactory.saveScreenshot(_, allowUniformRender))
  protected def currentSaveName: Option[String]
  protected def allowUniformRender: Boolean = false

