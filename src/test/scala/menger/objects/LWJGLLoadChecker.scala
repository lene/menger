package menger.objects

import com.badlogic.gdx.backends.lwjgl3.{Lwjgl3Application, Lwjgl3ApplicationConfiguration}
import com.badlogic.gdx.Gdx.gl20
import menger.MengerEngine

/**
 *  So here's the thing: when executing tests in sbt, the LWJGL library can be loaded only once.
 *  Subsequent invocations of the library will result in an UnsatisfiedLinkError.
 *  To work around this, we can check if the library can be loaded before running tests requiring
 *  LWJGL and use the result to conditionally run the tests, with `assume(loadingLWJGLSucceeds)`.
 */
object LWJGLLoadChecker:
  lazy val loadingLWJGLSucceeds: Boolean =
    try
      Lwjgl3Application(MengerEngine(0.01), Lwjgl3ApplicationConfiguration())
      true
    catch
      case _: UnsatisfiedLinkError => false
      case _: NullPointerException => false
