import com.github.sbt.jni.build.CMakeWithoutVersionBug

// Global settings applied to all projects
inThisBuild(List(
  semanticdbEnabled := true,
  semanticdbVersion := scalafixSemanticdb.revision,
  Test / parallelExecution := true,
))

lazy val mengerCommonDependency = "io.github.lene" %% "menger-common" % "0.1.1"
lazy val optixJniUri = uri("https://github.com/lene/optix-jni.git#c618caf")
lazy val optixJniProject =
  ProjectRef(optixJniUri, "optix-jni")

// Root project - aggregator only, no source code
lazy val root = project
  .in(file("."))
  .aggregate(mengerGeometry, mengerApp)
  .settings(
    name := "menger-root",
    publish / skip := true,
    cleanKeepFiles += (ThisBuild / baseDirectory).value / "target" / "compiler_plugins",
    // Delegate run to mengerApp so `sbt "run --args"` works from root
    Compile / run / aggregate := false,
    Compile / run := (mengerApp / Compile / run).evaluated,
    // Run all tests across all subprojects
    Test / test := {
      (mengerGeometry / Test / test).value
      (mengerApp / Test / test).value
    }
  )

// Menger-specific geometry layer — 4D fractals, caustics (not published)
lazy val mengerGeometry = project
  .in(file("menger-geometry"))
  .enablePlugins(JniNative)
  .dependsOn(optixJniProject)
  .settings(libraryDependencies += mengerCommonDependency)

// Main application - depends on menger-geometry and common
lazy val mengerApp = project
  .in(file("menger-app"))
  .dependsOn(mengerGeometry)
  .enablePlugins(JavaAppPackaging)
  .settings(
    libraryDependencies += mengerCommonDependency,
    // Set library path for run to find native library and CUDA libraries
    run / javaOptions += s"-Djava.library.path=${(mengerGeometry / target).value / "native" / "x86_64-linux" / "bin"}:/usr/local/cuda/lib64",
    run / fork := true,
    // Use project root as working directory so file paths match packaged executable behavior
    run / baseDirectory := (ThisBuild / baseDirectory).value
  )

Global / excludeLintKeys += root / cleanKeepFiles

// Temporary source-dependency bridge for Sprint 29.2:
// optix-jni has crossPaths := false, so its wartremover option points at
// target/compiler_plugins while Menger materializes the Scala 3 plugin under
// target/scala-2.12/compiler_plugins. Root clean preserves the copied directory
// because CI coverage runs `clean coverage test` in one sbt session. Remove this
// with the Sprint 29.6 artifact bump.
Global / onLoad := {
  val previous = (Global / onLoad).value
  state =>
    val loaded = previous(state)
    val extracted = Project.extract(loaded)
    val baseDir = extracted.get(ThisBuild / baseDirectory)
    extracted.get(mengerApp / Compile / scalacOptions)
    val mainCompilerPluginDir = baseDir / "target" / "scala-2.12" / "compiler_plugins"
    val sourceDependencyPluginDir = baseDir / "target" / "compiler_plugins"
    val wartremoverJars = (mainCompilerPluginDir * "wartremover_3.8.3-*.jar").get
    if (wartremoverJars.nonEmpty) {
      IO.createDirectory(sourceDependencyPluginDir)
      wartremoverJars.foreach { jar =>
        IO.copyFile(jar, sourceDependencyPluginDir / jar.getName)
      }
    }
    loaded
}
