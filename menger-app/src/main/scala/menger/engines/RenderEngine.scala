package menger.engines

trait RenderEngine:
  def create(): Unit
  def render(): Unit
  def resize(width: Int, height: Int): Unit
  def dispose(): Unit
  def pause(): Unit
  def resume(): Unit
  protected def onAnimationComplete(): Unit = ()
