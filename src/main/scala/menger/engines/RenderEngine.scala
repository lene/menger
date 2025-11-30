package menger.engines

import com.badlogic.gdx.Game

trait RenderEngine extends Game:
  // LibGDX lifecycle methods - concrete implementations required
  def create(): Unit
  def render(): Unit
  def resize(width: Int, height: Int): Unit
  def dispose(): Unit
  def pause(): Unit
  def resume(): Unit
