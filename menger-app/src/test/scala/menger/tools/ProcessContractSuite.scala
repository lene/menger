package menger.tools

import java.io.File
import java.nio.file.Files

import scala.sys.process.Process
import scala.sys.process.ProcessLogger

import menger.engines.WindowEvents
import menger.engines.WindowStateFile
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Every producer of the result the scene agent parses, and the window's events and state, held
  * to contracts/process-result.json -- one contract instead of each side encoding its own reading
  * (review 2026-10-14, menger#82, #100). The agent holds its parser to its copy of the file. */
class ProcessContractSuite extends AnyFlatSpec with Matchers:

  private val repoRoot: File =
    Iterator.iterate(new File(sys.props("user.dir")).getAbsoluteFile)(_.getParentFile)
      .takeWhile(_ != null) // scalafix:ok DisableSyntax.null
      .find(d => new File(d, "contracts/process-result.json").isFile)
      .getOrElse(fail("contracts/process-result.json not found above the working directory"))

  private val contract = ujson.read(Files.readString(new File(repoRoot, "contracts/process-result.json").toPath))

  private def conformsToContract(json: String): Unit =
    val result = ujson.read(json).obj
    contract("requiredFields").obj.foreach { (field, kind) =>
      withClue(s"$field in $json") {
        result.contains(field) shouldBe true
        typeOf(result(field)) shouldBe kind.str
      }
    }
    result("schemaVersion").str shouldBe contract("schemaVersion").str
    contract("exitCodes").obj.keySet should contain(result("tag").str)

  private def typeOf(value: ujson.Value): String = value match
    case _: ujson.Str => "string"
    case _: ujson.Arr => "array"
    case _: ujson.Num => "number"
    case _: ujson.Obj => "object"
    case _            => "other"

  "SceneValidator" should "use the contract's schema version and exit codes" in:
    SceneValidator.SchemaVersion shouldBe contract("schemaVersion").str
    SceneValidator.Tag.values.foreach { tag =>
      val wire = upickle.default.write(tag).stripPrefix("\"").stripSuffix("\"")
      withClue(wire) { SceneValidator.exitCodeFor(tag) shouldBe contract("exitCodes")(wire).num.toInt }
    }

  // menger-app's render-lock refusal (`menger.app.Main.refusedResultJson`) writes this same ValidationResult and exits with `exitCodeFor(Refused)`.
  it should "print a result that conforms to the contract" in:
    val refused = SceneValidator.ValidationResult(SceneValidator.Tag.Refused, List("usage"))
    conformsToContract(upickle.default.write(refused))

  private def runScript(script: String, args: String*): (Int, String) =
    val out = new StringBuilder
    val exit = Process(new File(repoRoot, s"docker/scene-validator/$script").getPath +: args)
      .!(ProcessLogger(line => out.append(line).append('\n'), _ => ()))
    (exit, out.toString)

  for script <- List("run-sandboxed.sh", "run-render-window.sh") do
    s"$script" should "refuse a missing scene with the contract's result and exit code" in:
      val args = if script == "run-render-window.sh" then List("--scene", "/nope/x.scala")
                 else List("/nope/x.scala")
      val (exit, stdout) = runScript(script, args*)
      conformsToContract(stdout.trim)
      ujson.read(stdout)("tag").str shouldBe "refused"
      exit shouldBe contract("exitCodes")("refused").num.toInt

  "The window's events and state" should "match the contract's kinds and versions" in:
    val events = contract("windowEvents")
    WindowEvents.Kind.values.map(_.wire).toSet shouldBe events("kinds").arr.map(_.str).toSet
    WindowEvents.Version shouldBe events("version").num.toInt
    val line = ujson.read(WindowEvents.line(WindowEvents.Kind.Up, "x")).obj
    line.keySet shouldBe events("fields").obj.keySet
    WindowStateFile.SchemaVersion shouldBe contract("windowStateSchemaVersion").str
