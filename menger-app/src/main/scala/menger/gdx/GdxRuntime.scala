package menger.gdx

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.InputProcessor

/**
 * Null-safe wrapper for LibGDX global singletons (Gdx.app, Gdx.graphics, Gdx.input, Gdx.gl).
 *
 * Consolidates all null-safe Gdx access in one place. All callers can use these
 * methods without explicit null guards or scalafix suppressions.
 */
object GdxRuntime:
  private def app      = Option(Gdx.app)
  private def graphics = Option(Gdx.graphics)
  private def input    = Option(Gdx.input)
  private def gl       = Option(Gdx.gl)

  def exit(): Unit                                  = app.foreach(_.exit())
  def requestRendering(): Unit                      = graphics.foreach(_.requestRendering())
  def setContinuousRendering(v: Boolean): Unit      = graphics.foreach(_.setContinuousRendering(v))
  def deltaTime: Float                              = graphics.map(_.getDeltaTime).getOrElse(0f)
  def width: Int                                    = graphics.map(_.getWidth).getOrElse(0)
  def height: Int                                   = graphics.map(_.getHeight).getOrElse(0)
  def glClear(mask: Int): Unit                      = gl.foreach(_.glClear(mask))
  def setInputProcessor(p: InputProcessor): Unit    = input.foreach(_.setInputProcessor(p))
  def isKeyPressed(keycode: Int): Boolean           = input.exists(_.isKeyPressed(keycode))
  def isButtonPressed(button: Int): Boolean         = input.exists(_.isButtonPressed(button))
