package menger.engines

trait SavesScreenshots:
  protected def saveImage(): Unit = currentSaveName.foreach(ScreenshotFactory.saveScreenshot)
  protected def currentSaveName: Option[String]

