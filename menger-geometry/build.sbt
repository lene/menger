import menger.build.CMakeWithoutVersionBug
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.jar.JarFile
import scala.jdk.CollectionConverters._
import scala.sys.process._

name := "menger-geometry"
version := "0.7.1"
scalaVersion := "3.8.3"

organization := "io.github.lene"
publish / skip := true   // internal module, not published separately

scalacOptions ++= Seq("-deprecation", "-explain", "-feature", "-Wunused:imports")

// Set library path for tests to find native libraries
Test / javaOptions ++= Seq(
  s"-Djava.library.path=${(Test / classDirectory).value / "native" / "x86_64-linux"}:${(Compile / classDirectory).value / "native" / "x86_64-linux"}:${target.value / "native" / "x86_64-linux" / "bin"}",
  "-Dlogback.statusListenerClass=ch.qos.logback.core.status.NopStatusListener"
)
Test / fork := true

lazy val optixJniNativeApiDir =
  settingKey[File]("Directory containing native API files extracted from optix-jni")
lazy val extractOptixJniNativeApi =
  taskKey[File]("Extract optix-jni native headers and shader includes from its published jar")
lazy val optixJniPinnedSource = OptixJniSource.projectRef
lazy val verifyOptixJniSourceCommit =
  taskKey[Unit]("Verify checked-out optix-jni source matches the pinned commit")

optixJniNativeApiDir := target.value / "optix-jni-native-api"

verifyOptixJniSourceCommit := {
  val log = streams.value.log
  val actual = Process(Seq("git", "rev-parse", "HEAD"), OptixJniSource.checkout).!!.trim
  if (actual != OptixJniSource.commit) {
    sys.error(
      s"optix-jni source checkout is at $actual, expected ${OptixJniSource.commit}"
    )
  }
  log.info(s"Verified optix-jni source commit ${actual.take(12)}")
}

def copyOptixJniNativeApiFromDirectory(nativeRoot: File, outputDir: File): Seq[File] =
  if (!nativeRoot.exists()) {
    Seq.empty
  } else {
    (nativeRoot ** "*").get.filter(_.isFile).map { sourceFile =>
      val relativePath = sourceFile.relativeTo(nativeRoot).fold(sourceFile.getName)(_.getPath)
      val targetFile = outputDir / relativePath
      IO.createDirectory(targetFile.getParentFile)
      IO.copyFile(sourceFile, targetFile)
      targetFile
    }
  }

def copyOptixJniNativeApiFromJar(optixJar: File, outputDir: File): Seq[File] = {
  val jar = new JarFile(optixJar)
  try {
    jar.entries.asScala
      .filter(entry => !entry.isDirectory && entry.getName.startsWith("optix-jni-native/"))
      .map { entry =>
        val relativePath = entry.getName.stripPrefix("optix-jni-native/")
        val targetFile = outputDir / relativePath
        IO.createDirectory(targetFile.getParentFile)
        val inputStream = jar.getInputStream(entry)
        try Files.copy(inputStream, targetFile.toPath, StandardCopyOption.REPLACE_EXISTING)
        finally inputStream.close()
        targetFile
      }
      .toSeq
  } finally {
    jar.close()
  }
}

extractOptixJniNativeApi := {
  val log = streams.value.log
  val outputDir = optixJniNativeApiDir.value

  IO.delete(outputDir)
  verifyOptixJniSourceCommit.value
  (optixJniPinnedSource / Compile / resources).value

  val classpathFiles = (Compile / dependencyClasspath).value.map(_.data)
  val optixJar = classpathFiles.find(_.getName.matches("optix-jni.*\\.jar"))
  val optixSourceNativeRoots = Seq(
    (optixJniPinnedSource / Compile / classDirectory).value / "optix-jni-native",
    (optixJniPinnedSource / Compile / resourceManaged).value / "optix-jni-native"
  )

  val extractedFiles = optixJar match {
    case Some(jarFile) =>
      copyOptixJniNativeApiFromJar(jarFile, outputDir)
    case None =>
      optixSourceNativeRoots
        .iterator
        .map(copyOptixJniNativeApiFromDirectory(_, outputDir))
        .find(_.nonEmpty)
        .getOrElse(Seq.empty)
  }

  if (extractedFiles.isEmpty) {
    val searched = (classpathFiles ++ optixSourceNativeRoots).map(_.getAbsolutePath).mkString(", ")
    sys.error(s"No optix-jni-native resources found. Searched: $searched")
  }

  log.info(s"Extracted ${extractedFiles.size} optix-jni native API files to $outputDir")
  outputDir
}

nativeCompile / sourceDirectory := sourceDirectory.value / "main" / "native"
nativeBuildTool := CMakeWithoutVersionBug.make(Seq(
  "-Wno-dev",
  "--log-level=WARNING",
  "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
  s"-DOPTIX_JNI_INCLUDE_DIR=${(optixJniNativeApiDir.value / "include").getAbsolutePath}",
  s"-DOPTIX_JNI_SHADER_DIR=${(optixJniNativeApiDir.value / "shaders").getAbsolutePath}"
))
nativeCompile := (nativeCompile dependsOn extractOptixJniNativeApi).value

// Bundle the menger PTX shader into the JAR as a managed resource.
// nativeCompile must run first (it produces the PTX via CMake).
Compile / resourceGenerators += Def.task {
  val log = streams.value.log
  val platform = "x86_64-linux"
  val ptxSource = target.value / "native" / platform / "bin" / "optix_shaders_menger.ptx"
  nativeCompile.value
  if (ptxSource.exists()) {
    val ptxResource = (Compile / resourceManaged).value / "native" / platform / "optix_shaders_menger.ptx"
    IO.copyFile(ptxSource, ptxResource)
    log.debug(s"Bundled menger PTX into managed resources: $ptxResource")
    Seq(ptxResource)
  } else {
    log.warn(s"Menger PTX not found after nativeCompile: $ptxSource")
    Seq.empty
  }
}.taskValue

libraryDependencies ++= Seq(
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.6",
  "ch.qos.logback" % "logback-classic" % "1.5.32",
  "org.scalatest" %% "scalatest" % "3.2.20" % Test
)
