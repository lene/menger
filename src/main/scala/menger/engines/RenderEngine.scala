package menger.engines

import com.badlogic.gdx.Game

/** Minimal interface for rendering engines.
  *
  * This trait defines the core LibGDX application lifecycle that all rendering
  * engines must implement, without imposing specific rendering technology choices.
  *
  * Concrete implementations include:
  * - MengerEngine: GDX-based mesh rendering for 3D/4D geometries
  * - OptiXEngine: GPU ray tracing via OptiX JNI
  */
trait RenderEngine extends Game:
  // LibGDX lifecycle methods - concrete implementations required
  def create(): Unit
  def render(): Unit
  def resize(width: Int, height: Int): Unit
  def dispose(): Unit
  def pause(): Unit
  def resume(): Unit
