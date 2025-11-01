package menger.objects

import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration
import menger.engines.InteractiveMengerEngine

/**
 *  So here's the thing: when executing tests in sbt, the LWJGL library can be loaded only once.
 *  Subsequent invocations of the library will result in an UnsatisfiedLinkError.
 *  To work around this, we can check if the library can be loaded before running tests requiring
 *  LWJGL and use the result to conditionally run the tests, with `assume(loadingLWJGLSucceeds)`.
 */
object LWJGLLoadChecker:
  given menger.ProfilingConfig = menger.ProfilingConfig.disabled
  lazy val loadingLWJGLSucceeds: Boolean =
    try
      Lwjgl3Application(InteractiveMengerEngine(timeout = 0.01), Lwjgl3ApplicationConfiguration())
      true
    catch
      case _: UnsatisfiedLinkError => false
      case _: NullPointerException => false
