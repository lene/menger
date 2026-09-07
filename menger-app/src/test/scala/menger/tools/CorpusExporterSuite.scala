package menger.tools

import java.nio.file.Files

import scala.jdk.CollectionConverters.IteratorHasAsScala

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import upickle.default.read

class CorpusExporterSuite extends AnyFlatSpec with Matchers:

  // CorpusExporter's own default source dir ("menger-app/src/main/scala/examples/dsl") is
  // relative to the repo root, matching `run / baseDirectory`'s override in build.sbt for
  // `sbt "mengerApp/runMain ..."` invocations. `Test / fork` has no such override, so sbt's
  // working directory for this suite is menger-app/ itself -- one path segment shorter.
  private val TestSourceDir = "src/main/scala/examples/dsl"

  private def freshTempPath(): String =
    val dir = Files.createTempDirectory("corpus-exporter-suite")
    dir.resolve("dsl-corpus.json").toString

  "CorpusExporter.resolveOutputPath" should "default to target/dsl-corpus.json when no args given" in:
    CorpusExporter.resolveOutputPath(Array.empty) shouldBe "target/dsl-corpus.json"

  it should "use the first argument when supplied" in:
    CorpusExporter.resolveOutputPath(Array("some/other/path.json")) shouldBe "some/other/path.json"

  "CorpusExporter.resolveSourceDir" should "default to examples/dsl when no second arg given" in:
    CorpusExporter.resolveSourceDir(Array("out.json")) shouldBe
      "menger-app/src/main/scala/examples/dsl"

  it should "use the second argument when supplied" in:
    CorpusExporter.resolveSourceDir(Array("out.json", "some/other/dir")) shouldBe "some/other/dir"

  "CorpusExporter.run" should "write a valid, versioned JSON corpus at the given path" in:
    val outputPath = freshTempPath()
    val result = CorpusExporter.run(Array(outputPath, TestSourceDir))

    result shouldBe Right(())
    Files.exists(java.nio.file.Paths.get(outputPath)) shouldBe true

    val json = Files.readString(java.nio.file.Paths.get(outputPath))
    val corpus = read[CorpusExporter.CorpusManifest](json)

    corpus.schemaVersion should not be empty
    corpus.scalaVersion shouldBe "3.8.3"
    corpus.optixJniVersion shouldBe "0.3.3"
    corpus.minDriverVersion shouldBe "580.65"

  it should "report a clear, non-empty error and not throw on an unwritable path" in:
    val blockingFile = Files.createTempFile("corpus-exporter-blocker", "")
    try
      val target = blockingFile.resolve("dsl-corpus.json").toString
      val result = CorpusExporter.run(Array(target, TestSourceDir))

      result.isLeft shouldBe true
      result.left.getOrElse("") should include(target)
    finally
      Files.deleteIfExists(blockingFile)

  it should "fail fast with a clear message when the source directory does not exist" in:
    val outputPath = freshTempPath()
    val result = CorpusExporter.run(Array(outputPath, "does/not/exist"))

    result.isLeft shouldBe true
    result.left.getOrElse("") should include("does/not/exist")

  "The generated corpus" should "exclude SceneIndex.scala (a registration index, not a scene)" in:
    val corpus = corpusFor(freshTempPath())
    corpus.scenes.map(_.name) should not contain "SceneIndex"

  it should "list only .scala files directly under the source directory" in:
    val corpus = corpusFor(freshTempPath())
    corpus.scenes.foreach(_.path should endWith(".scala"))

  it should "carry each scene's full source text" in:
    val corpus = corpusFor(freshTempPath())
    val simpleScene = corpus.scenes.find(_.name == "SimpleScene").getOrElse(
      fail("SimpleScene missing from generated corpus")
    )
    simpleScene.source should include("object SimpleScene")

  it should "carry every scene file currently present in examples/dsl" in:
    val corpus = corpusFor(freshTempPath())
    val expectedCount = java.nio.file.Files.list(
      java.nio.file.Paths.get(TestSourceDir)
    ).iterator().asScala
      .filter(p => Files.isRegularFile(p) && p.getFileName.toString.endsWith(".scala"))
      .count(p => p.getFileName.toString != "SceneIndex.scala")
    corpus.scenes should have size expectedCount

  private def corpusFor(outputPath: String): CorpusExporter.CorpusManifest =
    CorpusExporter.run(Array(outputPath, TestSourceDir)) shouldBe Right(())
    read[CorpusExporter.CorpusManifest](Files.readString(java.nio.file.Paths.get(outputPath)))
