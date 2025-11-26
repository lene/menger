package menger

/** Test utility that converts CLI parsing errors to exceptions for easier test assertions.
  *
  * The production MengerCLIOptions handles errors gracefully by printing help text and exiting.
  * In tests, we want parsing errors to throw exceptions so we can use `should be thrownBy`.
  */
class SafeMengerCLIOptions(args: Seq[String]) extends menger.MengerCLIOptions(args):
  @SuppressWarnings(Array("org.wartremover.warts.Throw"))
  override def onError(e: Throwable): Unit = throw e
