package menger.objects

import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration
import menger.engines.InteractiveMengerEngine


object LWJGLLoadChecker:
  given menger.ProfilingConfig = menger.ProfilingConfig.disabled
  lazy val loadingLWJGLSucceeds: Boolean =
    try
      Lwjgl3Application(InteractiveMengerEngine(timeout = 0.01), Lwjgl3ApplicationConfiguration())
      true
    catch
      case _: UnsatisfiedLinkError => false
      case _: NullPointerException => false
